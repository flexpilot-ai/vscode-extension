import { CoreMessage } from "ai";
import { basename } from "path";
import * as vscode from "vscode";
import { IChatResult } from "../interfaces";
import { logger } from "../logger";
import { Code, jsxToCoreMessage, Message } from "../prompts/jsx-utilities";
import { getEol, getLanguageConfig } from "../utilities";
import { VariablesManager } from "../variables";

/**
 * InlineChatPrompt class handles the generation of prompts for inline chat functionality.
 * It manages the context, history, and prompt generation for code editing scenarios.
 */
export class InlineChatPrompt {
  private readonly history: { request: string; response: string }[];
  public readonly editor: vscode.TextEditor;
  private readonly participant = "flexpilot.editor.default";

  /**
   * Constructs an InlineChatPrompt instance.
   * @param {vscode.ChatResponseStream} response - The chat response stream.
   * @param {vscode.ChatContext} context - The chat context.
   * @param {vscode.ChatRequest} request - The chat request.
   * @throws {Error} If no active text editor is found.
   */
  constructor(
    private readonly response: vscode.ChatResponseStream,
    context: vscode.ChatContext,
    private readonly request: vscode.ChatRequest,
  ) {
    logger.info("Initializing InlineChatPrompt");
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      throw new Error("No active text editor found for Inline Chat");
    }
    this.editor = editor;
    this.expandSelectionToLineBoundary();
    this.history = context.history
      .filter((item) => item.participant === this.participant)
      .filter((item) => item instanceof vscode.ChatResponseTurn)
      .map((item) => (item.result as IChatResult).metadata);
    logger.debug(
      `InlineChatPrompt initialized with ${this.history.length} history items`,
    );
  }

  /**
   * Expands the current selection to full lines.
   */
  private readonly expandSelectionToLineBoundary = () => {
    const start = this.editor.selection.start.with({ character: 0 });
    let end = this.editor.selection.end;
    if (end.character !== 0) {
      end = new vscode.Position(end.line + 1, 0);
    }
    this.editor.selection = new vscode.Selection(start, end);
    logger.debug(
      `Selection expanded: ${start.line}:${start.character} to ${end.line}:${end.character}`,
    );
  };

  /**
   * Generates the system prompt.
   * @returns {CoreMessage} The system prompt as a CoreMessage.
   */
  private readonly getSystemPrompt = (): CoreMessage => {
    logger.debug("Generating system prompt");
    return jsxToCoreMessage(
      <Message role="system">
        <ul>
          <li>
            You're a skilled programmer named "Flexpilot" assisting a fellow
            developer in revising a code snippet.
          </li>
          <li>
            Your colleague will provide you with a file and a specific section
            to modify, along with a set of guidelines. Kindly rewrite the
            selected code in accordance with their instructions.
          </li>
          <li>
            The user is operating on a `{process.platform}` system. Please use
            system-specific commands when applicable.
          </li>
          <li>
            Carefully consider and analyze the rewrite to ensure it best aligns
            with their instructions.
          </li>
          <li>
            The active file or document is the source code the user is looking
            at right now.
          </li>
        </ul>
      </Message>,
    );
  };

  /**
   * Generates code boundary markers based on the current language.
   * @returns {{ start: string; end: string }} The start and end boundary markers.
   */
  public getCodeBoundary(): { start: string; end: string } {
    const { comment } = getLanguageConfig(this.editor.document.languageId);
    return {
      start: `${comment.start}Start of Selection${comment.end}`,
      end: `${comment.start}End of Selection${comment.end}`,
    };
  }

  /**
   * Generates the context prompt including the current file content and selection.
   * @returns {CoreMessage} The context prompt as a CoreMessage.
   */
  private getContextPrompt(): CoreMessage {
    logger.debug("Generating context prompt");
    const { document, selection } = this.editor;
    const fullRange = document.validateRange(
      new vscode.Range(0, 0, document.lineCount, 0),
    );
    const codeBoundary = this.getCodeBoundary();
    const content = [
      document.getText(fullRange.with({ end: selection.start })),
      codeBoundary.start,
      getEol(document),
      document.getText(selection),
      codeBoundary.end,
      getEol(document),
      document.getText(fullRange.with({ start: selection.end })),
    ].join("");
    return jsxToCoreMessage(
      <Message role="user">
        <p>
          Here's the content from active file with the selected area
          highlighted:
        </p>
        <pre>
          <code>{content}</code>
        </pre>
        <p>The active file is located in {basename(document.fileName)}.</p>
      </Message>,
    );
  }

  /**
   * Generates the edit prompt with instructions for code rewriting.
   * @returns {CoreMessage} The edit prompt as a CoreMessage.
   */
  private getEditPrompt(): CoreMessage {
    logger.debug("Generating edit prompt");
    const { comment, markdown } = getLanguageConfig(
      this.editor.document.languageId,
    );
    const codeBoundary = this.getCodeBoundary();
    return jsxToCoreMessage(
      <Message role="user">
        <p>Kindly rewrite this selection based on the following guidelines:</p>
        <h2>Editing Instructions</h2>
        <p>
          {this.history.length
            ? this.history.map((item) => item.request).join(" ,")
            : this.request.prompt}
        </p>
        <h2>Code Section to Modify</h2>
        <Code language={markdown}>
          {this.editor.document.getText(this.editor.selection)}
        </Code>
        Please rewrite the highlighted code according to the provided
        instructions. Remember to only modify the code within the selected area.
        <p>
          Please structure your response as follows:
          <Code language={markdown}>
            {codeBoundary.start}
            {"\n"}
            {comment.start}PUT_YOUR_REWRITE_HERE{comment.end}
            {"\n"}
            {codeBoundary.end}
          </Code>
        </p>
        Begin your response immediately with ```
      </Message>,
    );
  }

  /**
   * Generates the follow-up prompt for subsequent edits.
   * @returns {CoreMessage} The follow-up prompt as a CoreMessage.
   */
  private getFollowUpPrompt(): CoreMessage {
    logger.debug("Generating follow-up prompt");
    return jsxToCoreMessage(
      <Message role="user">
        <h2>Guidelines</h2>
        <section>
          <h3>Initial Request</h3>
          <p>{this.history.map((item) => item.request).join(" ,")}</p>
        </section>
        <section>
          <h3>Additional Request</h3>
          <p>{this.request.prompt}</p>
        </section>
        <p>
          Please rewrite your edit in accordance with the additional
          instructions. Once again, rewrite the entire selected section.
        </p>
      </Message>,
    );
  }

  /**
   * Builds the complete set of messages for the chat prompt.
   * @returns {Promise<CoreMessage[]>} A promise that resolves to an array of CoreMessages.
   */
  public async build(): Promise<CoreMessage[]> {
    logger.info("Building chat prompt messages");
    let messages: CoreMessage[] = [this.getSystemPrompt()];
    const references = await VariablesManager.resolveVariablesToCoreMessages(
      this.request,
      this.response,
    );
    messages = messages.concat(references);
    messages.push(this.getContextPrompt(), this.getEditPrompt());
    const lastMessage = this.history.pop();
    if (lastMessage) {
      messages.push({ role: "assistant", content: lastMessage.response });
      messages.push(this.getFollowUpPrompt());
    }
    logger.debug(`Built ${messages.length} messages for chat prompt`);
    return messages;
  }
}
