import { CoreMessage, generateText, streamText } from "ai";
import * as vscode from "vscode";
import { IChatResult } from "./interfaces";
import { logger } from "./logger";
import { PanelChatPrompt } from "./prompts/panel-chat";
import { ModelProviderManager } from "./providers";
import { storage } from "./storage";
import { VariablesManager } from "./variables";

/**
 * PanelChatParticipant class provides functionality for the panel chat feature in the Flexpilot.
 * It implements the Singleton pattern to ensure a single instance across the application.
 */
class PanelChatParticipant {
  private static instance: PanelChatParticipant | null = null;
  private readonly chatParticipant: vscode.ChatParticipant;
  private readonly githubSession: vscode.AuthenticationSession;

  /**
   * Private constructor to prevent direct instantiation.
   * Initializes the chat participant with necessary providers and configurations.
   */
  private constructor() {
    // Get the GitHub session
    this.githubSession = storage().session.get();

    // Create the chat participant
    this.chatParticipant = vscode.chat.createChatParticipant(
      "flexpilot.panel.default",
      this.handleChatRequest.bind(this),
    );

    // Set up welcome message and sample questions providers
    this.chatParticipant.welcomeMessageProvider = {
      provideWelcomeMessage: this.provideWelcomeMessage.bind(this),
      provideSampleQuestions: this.provideSampleQuestions.bind(this),
    };

    // Set up title provider
    this.chatParticipant.titleProvider = {
      provideChatTitle: this.provideChatTitle.bind(this),
    };

    // Set up followup provider
    this.chatParticipant.followupProvider = {
      provideFollowups: this.provideFollowups.bind(this),
    };

    // Configure help text prefix
    this.chatParticipant.helpTextPrefix = PanelChatPrompt.getHelpTextPrefix();
    this.chatParticipant.helpTextPrefix.isTrusted = {
      enabledCommands: ["flexpilot.configureModel", "flexpilot.viewLogs"],
    };

    // Configure help text postfix
    this.chatParticipant.helpTextPostfix = PanelChatPrompt.getHelpTextPostfix();
    this.chatParticipant.helpTextPostfix.isTrusted = {
      enabledCommands: ["inlineChat.start"],
    };

    // Set chat participant icon
    this.chatParticipant.iconPath = new vscode.ThemeIcon("flexpilot-default");

    // Set up requester information
    this.chatParticipant.requester = {
      name: this.githubSession.account.label,
      icon: vscode.Uri.parse(
        `https://avatars.githubusercontent.com/u/${this.githubSession.account.id}`,
      ),
    };

    // Set up help text variables prefix
    this.chatParticipant.helpTextVariablesPrefix =
      PanelChatPrompt.getHelpTextVariablesPrefix();
  }

  /**
   * Disposes the panel chat participant instance.
   */
  public static dispose(): void {
    if (PanelChatParticipant.instance) {
      PanelChatParticipant.instance.chatParticipant.dispose();
      PanelChatParticipant.instance = null;
    }
    logger.info("Panel chat participant disposed successfully");
  }

  /**
   * Registers the panel chat participant instance.
   */
  public static register() {
    if (!PanelChatParticipant.instance) {
      PanelChatParticipant.instance = new PanelChatParticipant();
      logger.debug("Panel chat participant registered successfully");
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
    try {
      // Create an abort controller for the chat request
      const abortController = new AbortController();
      token.onCancellationRequested(() => {
        abortController.abort();
      });

      // Get the chat provider
      const provider =
        ModelProviderManager.getInstance().getProvider<"chat">("Panel Chat");
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
            message: `Model not configured for \`Panel Chat\``,
          },
        };
      }

      // Prepare messages for the chat
      let messages: CoreMessage[] = [
        PanelChatPrompt.getChatProviderSystemPrompt(provider.config.model),
      ];
      // Get history prompts
      messages = messages.concat(PanelChatPrompt.getHistoryPrompts(context));
      // Resolve variables to core messages
      messages = messages.concat(
        await VariablesManager.resolveVariablesToCoreMessages(
          request,
          response,
        ),
      );
      // Add user prompt
      messages.push({ role: "user", content: request.prompt });

      // Generate the chat response
      const stream = await streamText({
        model: await provider.model(),
        messages: messages,
        abortSignal: abortController.signal,
        stopSequences: [],
        temperature: storage().workspace.get<number>(
          "flexpilot.panelChat.temperature",
        ),
      });

      // Stream the response
      for await (const textPart of stream.fullStream) {
        // if part is not of text-delta skip
        if (textPart.type !== "text-delta") continue;

        // stream response to chat panel
        response.markdown(textPart.textDelta);
      }

      // Check if token usage is enabled and show usage
      if (storage().workspace.get("flexpilot.panelChat.showTokenUsage")) {
        const usage = await stream.usage;
        if (usage.completionTokens && usage.promptTokens) {
          response.warning(
            `Prompt Tokens: ${usage.promptTokens}, Completion Tokens: ${usage.completionTokens}`,
          );
        }
      }

      // Set the context to indicate chat for walkthroughs
      await vscode.commands.executeCommand(
        "setContext",
        "flexpilot:walkthroughPanelChat",
        true,
      );

      // Return the chat result
      logger.debug(`Model Response: ${await stream.text}`);
      return {
        metadata: {
          response: await stream.text,
          request: request.prompt,
        },
      };
    } catch (error) {
      // Log and return error response
      logger.error(error as Error);
      logger.notifyError("Error processing `Panel Chat` request");
      return {
        metadata: {
          response: "Unable to process request",
          request: request.prompt,
        },
        errorDetails: { message: "Error processing request" },
      };
    }
  }

  /**
   * Provides the welcome message for the chat.
   * @returns {Promise<vscode.ChatWelcomeMessageContent>} The welcome message content.
   */
  private async provideWelcomeMessage(): Promise<vscode.ChatWelcomeMessageContent> {
    return {
      icon: new vscode.ThemeIcon("flexpilot-default"),
      title: "Ask Flexpilot",
      message: PanelChatPrompt.getWelcomeMessage(
        this.githubSession.account.label,
      ),
    };
  }

  /**
   * Provides sample questions for the chat.
   * @param {vscode.ChatLocation} location - The chat location.
   * @param {vscode.CancellationToken} token - The cancellation token.
   * @returns {Promise<vscode.ChatFollowup[]>} An array of sample questions.
   */
  private async provideSampleQuestions(
    location: vscode.ChatLocation,
    token: vscode.CancellationToken,
  ): Promise<vscode.ChatFollowup[]> {
    return location === vscode.ChatLocation.Panel &&
      !token.isCancellationRequested
      ? [{ prompt: "/help - Get help with Flexpilot commands" }]
      : [];
  }

  /**
   * Provides a title for the chat based on the context.
   * @param {vscode.ChatContext} context - The chat context.
   * @param {vscode.CancellationToken} token - The cancellation token.
   * @returns {Promise<string>} The chat title.
   */
  private async provideChatTitle(
    context: vscode.ChatContext,
    token: vscode.CancellationToken,
  ): Promise<string> {
    // Default chat title
    let defaultTitle;
    if (context.history[0] instanceof vscode.ChatRequestTurn) {
      defaultTitle = context.history[0].prompt;
    } else {
      return "Untitled Chat";
    }
    try {
      // Create an abort controller for the chat title
      const abortController = new AbortController();
      token.onCancellationRequested(() => {
        abortController.abort();
      });

      // Get the chat provider for chat title
      const provider =
        ModelProviderManager.getInstance().getProvider<"chat">("Chat Title");
      if (!provider) {
        return defaultTitle;
      }

      // Prepare messages for chat title
      const messages: CoreMessage[] = [
        PanelChatPrompt.getTitleProviderSystemPrompt(),
        { role: "user", content: defaultTitle },
        PanelChatPrompt.getTitleProviderUserPrompt(),
      ];

      // Generate the chat title
      const response = await generateText({
        model: await provider.model(),
        messages: messages,
        abortSignal: abortController.signal,
        stopSequences: [],
        temperature: storage().workspace.get<number>(
          "flexpilot.chatTitle.temperature",
        ),
      });

      // Return the chat title
      logger.debug(`Model Response: ${response.text}`);
      return response.text.trim();
    } catch (error) {
      logger.error(error as Error);
      logger.notifyError("Error processing `Chat Title` request");
      return defaultTitle;
    }
  }

  /**
   * Provides followup suggestions for the chat.
   * @param {IChatResult} result - The chat result.
   * @param {vscode.ChatContext} context - The chat context.
   * @param {vscode.CancellationToken} token - The cancellation token.
   * @returns {Promise<vscode.ChatFollowup[]>} An array of followup suggestions.
   */
  private async provideFollowups(
    result: IChatResult,
    context: vscode.ChatContext,
    token: vscode.CancellationToken,
  ): Promise<vscode.ChatFollowup[]> {
    try {
      // Check if the result contains metadata
      if (!result.metadata?.response || !result.metadata?.request) {
        return [];
      }

      // Create an abort controller for the followup suggestions
      const abortController = new AbortController();
      token.onCancellationRequested(() => {
        abortController.abort();
      });

      // Get the chat provider for suggestions
      const provider =
        ModelProviderManager.getInstance().getProvider<"chat">(
          "Chat Suggestions",
        );
      if (!provider) {
        return [{ prompt: "Model not configured for Chat Suggestions" }];
      }

      // Prepare messages for followup suggestions
      let messages: CoreMessage[] = [
        PanelChatPrompt.getFollowUpProviderSystemPrompt(),
      ];
      messages = messages.concat(PanelChatPrompt.getHistoryPrompts(context));
      messages.push(
        { role: "user", content: result.metadata.request },
        { role: "assistant", content: result.metadata.response },
        PanelChatPrompt.getFollowUpProviderUserPrompt(),
      );

      // Generate the followup suggestions
      const response = await generateText({
        model: await provider.model(),
        messages: messages,
        abortSignal: abortController.signal,
        stopSequences: [],
        temperature: storage().workspace.get<number>(
          "flexpilot.chatSuggestions.temperature",
        ),
      });

      // Return the followup suggestions
      logger.debug(`Model Response: ${response.text}`);
      return [{ prompt: response.text.trim() }];
    } catch (error) {
      logger.error(error as Error);
      logger.notifyError("Error processing `Chat Suggestions` request");
      return [{ prompt: "Error generating followups" }];
    }
  }
}

// Export the PanelChatParticipant class
export default PanelChatParticipant;
