import { streamText } from "ai";
import * as vscode from "vscode";
import { IChatResult } from "./interfaces";
import { logger } from "./logger";
import { InlineChatPrompt } from "./prompts/inline-chat";
import { ModelProviderManager } from "./providers";
import { storage } from "./storage";
import { getEol } from "./utilities";

/**
 * InlineChatParticipant class provides functionality for the inline chat feature in Flexpilot.
 * It implements the Singleton pattern to ensure a single instance across the application.
 */
class InlineChatParticipant {
  private static instance: InlineChatParticipant | null = null;
  private readonly chatParticipant: vscode.ChatParticipant;
  private readonly githubSession: vscode.AuthenticationSession;

  /**
   * Private constructor to prevent direct instantiation.
   * Initializes the chat participant with necessary providers and configurations.
   */
  private constructor() {
    // Create the chat participant
    this.chatParticipant = vscode.chat.createChatParticipant(
      "flexpilot.editor.default",
      this.handleChatRequest.bind(this),
    );

    // Get the GitHub session
    this.githubSession = storage().session.get();

    // Set up requester information
    this.chatParticipant.requester = {
      name: this.githubSession.account.label,
      icon: vscode.Uri.parse(
        `https://avatars.githubusercontent.com/u/${this.githubSession.account.id}`,
      ),
    };

    // Set chat participant icon
    this.chatParticipant.iconPath = new vscode.ThemeIcon("flexpilot-default");
  }

  /**
   * Disposes the inline chat participant instance.
   */
  public static dispose(): void {
    if (InlineChatParticipant.instance) {
      InlineChatParticipant.instance.chatParticipant.dispose();
      InlineChatParticipant.instance = null;
    }
    logger.info("Inline chat participant disposed successfully");
  }

  /**
   * Registers the inline chat participant instance.
   */
  public static register() {
    if (!InlineChatParticipant.instance) {
      InlineChatParticipant.instance = new InlineChatParticipant();
      logger.debug("Inline chat participant registered successfully");
    }
  }

  /**
   * Handles the chat request and generates a response.
   * @param {vscode.ChatRequest} request - The chat request.
   * @param {vscode.ChatContext} context - The chat context.
   * @param {vscode.ChatResponseStream} response - The response stream.
   * @param {vscode.CancellationToken} token - The cancellation token.
   * @returns {Promise<IChatResult>} The chat result.
   */
  private async handleChatRequest(
    request: vscode.ChatRequest,
    context: vscode.ChatContext,
    response: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
  ): Promise<IChatResult> {
    const abortController = new AbortController();
    token.onCancellationRequested(() => {
      abortController.abort();
    });

    try {
      // Get the chat provider
      const provider =
        ModelProviderManager.getInstance().getProvider<"chat">("Inline Chat");
      if (!provider) {
        response.markdown("Click below button to configure model");
        response.button({
          command: "flexpilot.configureModel",
          title: "Configure Model",
        });
        return {
          metadata: {
            response: "Unable to process request",
            request: request.prompt,
          },
          errorDetails: {
            message: `Model not configured for \`Inline Chat\``,
          },
        };
      }

      // Prepare messages for the chat
      const prompt = new InlineChatPrompt(response, context, request);
      const messages = await prompt.build();

      // Get the document and selection
      const { document, selection } = prompt.editor;
      const codeBoundary = prompt.getCodeBoundary();

      // Generate the chat response
      const stream = await streamText({
        model: await provider.model(),
        messages: messages,
        abortSignal: abortController.signal,
        stopSequences: [codeBoundary.end],
        temperature: storage().workspace.get<number>(
          "flexpilot.inlineChat.temperature",
        ),
      });

      let lastPushedIndex = 0;
      let final = "";

      for await (const textPart of stream.fullStream) {
        // if part is not of text-delta skip
        if (textPart.type !== "text-delta") continue;

        // append the text part to the final response
        final = final.concat(textPart.textDelta);
        const lines = final.split("\n");

        // get index of start and end boundary
        const trimmed = lines.map((line) => line.trim());
        const startIdx = trimmed.indexOf(codeBoundary.start);
        let endIdx = trimmed.length;
        if (trimmed.includes(codeBoundary.end)) {
          endIdx = trimmed.indexOf(codeBoundary.end);
        }

        // continue if the start boundary is not found
        if (startIdx < 0) {
          continue;
        }

        // get the filtered lines between boundary
        const filtered = lines.slice(startIdx + 1, endIdx + 1);

        // push the filtered lines to the response
        for (let i = lastPushedIndex; i < filtered.length - 1; i++) {
          if (i === 0) {
            // replace the selection with the first line
            response.push(
              new vscode.ChatResponseTextEditPart(document.uri, {
                range: selection,
                newText: filtered[i] + getEol(document),
              }),
            );
          } else {
            // insert the rest of the lines after the first line
            response.push(
              new vscode.ChatResponseTextEditPart(document.uri, {
                range: new vscode.Range(
                  selection.start.translate(i),
                  selection.start.translate(i),
                ),
                newText: filtered[i] + getEol(document),
              }),
            );
          }
          lastPushedIndex = i + 1;
        }

        // break if the end boundary is found
        if (trimmed.includes(codeBoundary.end)) {
          break;
        }
      }

      // Check if token usage is enabled and show usage
      if (storage().workspace.get("flexpilot.inlineChat.showTokenUsage")) {
        const usage = await stream.usage;
        if (usage.completionTokens && usage.promptTokens) {
          response.warning(
            `Prompt Tokens: ${usage.promptTokens}, Completion Tokens: ${usage.completionTokens}`,
          );
        }
      }

      // Log the model response
      logger.debug(`Model Response: ${await stream.text}`);
      return { metadata: { response: final, request: request.prompt } };
    } catch (error) {
      logger.error(error as Error);
      logger.notifyError("Error processing `Inline Chat` request");
      throw error;
    }
  }
}

// Export the InlineChatParticipant instance
export default InlineChatParticipant;
