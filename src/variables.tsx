import { CoreUserMessage } from "ai";
import { basename } from "path";
import * as vscode from "vscode";
import { eventsManager } from "./events";
import { logger } from "./logger";
import {
  Code,
  jsxToCoreMessage,
  jsxToMarkdown,
  Message,
} from "./prompts/jsx-utilities";
import { getLanguageConfig, getTerminalType } from "./utilities";

/**
 * VariablesManager class provides centralized management of chat variables and prompts.
 * It implements the Singleton pattern to ensure a single instance across the application.
 */
export class VariablesManager {
  private static instance: VariablesManager;
  private readonly disposables: vscode.Disposable[] = [];

  private constructor() {
    const chatVariables = [
      {
        name: "editor",
        description: "The visible source code in the active editor",
        isSlow: false,
        fullName: "Current Editor",
        icon: new vscode.ThemeIcon("file"),
        prompt: this.getEditorPrompt,
      },
      {
        name: "selection",
        description: "The current selection in the active editor",
        isSlow: false,
        fullName: "Active Selection",
        icon: new vscode.ThemeIcon("selection"),
        prompt: this.getSelectionPrompt,
      },
      {
        name: "terminalLastCommand",
        description: "The last command executed in the terminal",
        isSlow: false,
        fullName: "Terminal Last Command",
        icon: new vscode.ThemeIcon("terminal"),
        prompt: this.getTerminalLastCommandPrompt,
      },
      {
        name: "terminalSelection",
        description: "The selected text from terminal",
        isSlow: false,
        fullName: "Terminal Selection",
        icon: new vscode.ThemeIcon("terminal"),
        prompt: this.getTerminalSelectionPrompt,
      },
    ];
    logger.info("Registering chat variable resolvers");
    for (const variable of chatVariables) {
      this.disposables.push(
        vscode.chat.registerChatVariableResolver(
          `flexpilot.${variable.name}`,
          variable.name,
          variable.description,
          variable.description,
          variable.isSlow,
          {
            resolve2: (_name, _context, stream) => {
              try {
                logger.debug(`Resolving variable: ${variable.name}`);
                const prompt = variable.prompt();
                if (prompt.props.reference) {
                  stream.reference(prompt.props.reference);
                }
                const message = jsxToMarkdown(prompt);
                return [
                  {
                    level: vscode.ChatVariableLevel.Full,
                    value: message.value,
                    description: variable.description,
                  },
                ];
              } catch (error) {
                logger.warn(`Chat Variable Resolution Error: ${error}`);
                return [];
              }
            },
            resolve: () => [],
          },
          variable.fullName,
          variable.icon,
        ),
      );
    }
    logger.info("Chat variable resolvers registered");
  }

  /**
   * Registers the VariablesManager instance.
   */
  public static register() {
    if (!VariablesManager.instance) {
      VariablesManager.instance = new VariablesManager();
      logger.debug("New VariablesManager instance created");
    }
  }

  /**
   * Disposes of all registered disposables.
   */
  public dispose(): void {
    this.disposables.forEach((item) => item.dispose());
    logger.info("VariablesManager disposed");
  }

  /**
   * Get the files prompt for a given file or location.
   * @param {vscode.Uri | vscode.Location} file - The file or location to get the prompt for.
   * @returns {Promise<JSX.Element>} A promise that resolves to the JSX element for the files prompt.
   */
  private static async getFilesPrompt(
    file: vscode.Uri | vscode.Location,
  ): Promise<JSX.Element> {
    const uri = file instanceof vscode.Location ? file.uri : file;
    logger.info(`Getting files prompt for: ${uri.toString()}`);
    const document = await vscode.workspace.openTextDocument(uri);
    const range =
      file instanceof vscode.Location
        ? file.range
        : new vscode.Range(0, 0, document.lineCount + 1, 0);
    const language = getLanguageConfig(document.languageId);
    const path = document.uri.fsPath;
    const startLine = range.start.line + 1;
    const endLine = range.end.line + 1;
    const content = document.getText(range).trim();
    logger.debug(`Files prompt generated for: ${path}`);
    return (
      <Message role="user" reference={document.uri}>
        <h1>FILE:${basename(uri.fsPath)} CONTEXT</h1>
        <br />
        Use below file content for reference
        <br />
        <br />
        (file: <code>{path}</code>, lines {startLine} to {endLine})
        <br />
        <Code language={language.markdown}>{content.trim()}</Code>
      </Message>
    );
  }

  /**
   * Retrieves the current active editor's prompt as a JSX element.
   * Logs the process of obtaining the editor prompt.
   * If no active editor is found, logs a warning and returns undefined.
   *
   * @returns {JSX.Element} A JSX element containing the editor context or undefined if no active editor is found.
   */
  private getEditorPrompt(): JSX.Element {
    logger.info("Getting editor prompt");
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
      throw new Error("No active editor found");
    }
    const language = getLanguageConfig(activeEditor.document.languageId);
    const path = activeEditor.document.uri.fsPath;
    const lines = activeEditor.document.lineCount;
    const content = activeEditor.document.getText();
    logger.debug(`Editor prompt generated for: ${path}`);
    return (
      <Message role="user" reference={activeEditor.document.uri}>
        <h1>EDITOR CONTEXT</h1>
        <br />
        Use below content from active editor for reference
        <br />
        <br />
        (file: <code>{path}</code>, lines 1 to {lines})
        <br />
        <Code language={language.markdown}>{content.trim()}</Code>
      </Message>
    );
  }

  /**
   * Retrieves the current active editor's selection prompt as a JSX element.
   * Logs the process of obtaining the selection prompt.
   * If no active editor is found, logs a warning and returns undefined.
   *
   * @returns {JSX.Element} A JSX element containing the selection context or undefined if no active editor is found.
   */
  private getSelectionPrompt(): JSX.Element {
    logger.info("Getting selection prompt");
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
      throw new Error("No active editor found with selection");
    }
    const { document, selection } = activeEditor;
    const language = getLanguageConfig(document.languageId);
    const startLine = selection.start.line + 1;
    const endLine = selection.end.line + 1;
    const path = document.uri.fsPath;
    const content = document.getText(selection);
    logger.debug(`Selection prompt generated for: ${path}`);
    return (
      <Message
        role="user"
        reference={new vscode.Location(document.uri, selection)}
      >
        <h1>SELECTION CONTEXT</h1>
        <br />
        Use below content selection from active editor for reference
        <br />
        <br />
        (file: <code>{path}</code>, lines {startLine} to {endLine})
        <br />
        <Code language={language.markdown}>{content.trim()}</Code>
      </Message>
    );
  }

  /**
   * Retrieves the terminal selection prompt as a JSX element.
   * Logs the process of obtaining the terminal selection prompt.
   * If no active terminal or selection is found, logs a warning and returns undefined.
   *
   * @returns {JSX.Element} A JSX element containing the terminal selection context or undefined if no active terminal or selection is found.
   */
  private getTerminalSelectionPrompt(): JSX.Element {
    logger.info("Getting terminal selection prompt");
    const terminal = vscode.window.activeTerminal;
    if (!terminal?.selection) {
      throw new Error("No active terminal selection found");
    }
    logger.debug("Terminal selection prompt generated");
    return (
      <Message role="user">
        <h1>TERMINAL SELECTION CONTEXT</h1>
        <br />
        Use below content from active terminal selection for reference
        <br />
        <Code language={getTerminalType(terminal)}>
          {terminal.selection.trim()}
        </Code>
      </Message>
    );
  }

  /**
   * Retrieves the terminal last command prompt as a JSX element.
   * Logs the process of obtaining the terminal last command prompt.
   * If no last terminal shell execution is found, logs a warning and returns undefined.
   *
   * @returns {JSX.Element} A JSX element containing the terminal last command context or undefined if no last terminal shell execution is found.
   */
  private getTerminalLastCommandPrompt(): JSX.Element {
    logger.info("Getting terminal last command prompt");
    const event = eventsManager().getLastTerminalExecutedCommand();
    if (!event) {
      throw new Error("No last terminal shell execution found");
    }
    const terminalType = getTerminalType(event.terminal);
    logger.debug("Terminal last command prompt generated");
    return (
      <Message role="user">
        <h1>TERMINAL LAST COMMAND CONTEXT</h1>
        {event.commandLine && (
          <>
            <br />
            The following is the last command run in the terminal:
            <br />
            <Code language={terminalType}>{event.commandLine}</Code>
          </>
        )}
        {event.cwd && (
          <>
            <br />
            It was run in the directory:
            <br />
            <Code language={terminalType}>{event.cwd.toString()}</Code>
          </>
        )}
        {event.terminal && (
          <>
            <br />
            It was run using shell type:
            <br />
            <Code language={terminalType}>{terminalType}</Code>
          </>
        )}
        {event.output && (
          <>
            <br />
            It has the following output:
            <br />
            <Code language={terminalType}>{event.output}</Code>
          </>
        )}
      </Message>
    );
  }

  /**
   * Resolves variables to core messages.
   * @param {vscode.ChatRequest} request - The chat request.
   * @param {vscode.ChatResponseStream} response - The chat response stream.
   * @returns {Promise<CoreUserMessage[]>} A promise that resolves to an array of core user messages.
   */
  public static async resolveVariablesToCoreMessages(
    request: vscode.ChatRequest,
    response: vscode.ChatResponseStream,
  ): Promise<CoreUserMessage[]> {
    const messages: CoreUserMessage[] = [];
    for (const reference of request.references) {
      try {
        if (typeof reference.value === "string" && reference.id) {
          response.reference2({ variableName: reference.name });
          messages.push({ role: "user", content: reference.value });
        } else if (
          reference.value instanceof vscode.Uri ||
          reference.value instanceof vscode.Location
        ) {
          const prompt = await this.getFilesPrompt(reference.value);
          if (prompt.props.reference) {
            response.reference2(prompt.props.reference);
          }
          messages.push(jsxToCoreMessage(prompt) as CoreUserMessage);
        }
      } catch (error) {
        logger.warn(`Chat Variable Resolution Error: ${error}`);
      }
    }
    return messages;
  }
}
