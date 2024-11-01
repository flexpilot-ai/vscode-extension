import { createAnthropic } from "@ai-sdk/anthropic";
import { generateText, LanguageModelV1 } from "ai";
import * as vscode from "vscode";
import { IChatModelProvider, IModelConfig } from "../interfaces";
import { logger } from "../logger";
import { storage } from "../storage";

/**
 * Extended interface for Anthropic chat model configuration
 */
interface IAnthropicChatModelConfig extends IModelConfig {
  apiKey: string;
  baseUrl: string;
}

/**
 * Default help prompt for Anthropic API key configuration
 */
const DEFAULT_HELP_PROMPT =
  "Click [here](https://docs.flexpilot.ai/model-providers/anthropic.html) for more information";

/**
 * Anthropic Chat Model Provider class
 * Manages the configuration and creation of Anthropic chat models
 */
export class AnthropicChatModelProvider extends IChatModelProvider {
  static readonly providerName = "Anthropic";
  static readonly providerId = "anthropic-chat";
  static readonly providerType = "chat" as const;
  public readonly config: IAnthropicChatModelConfig;

  /**
   * Constructor for AnthropicChatModelProvider
   * @param {string} nickname - The nickname for the model configuration
   * @throws {Error} If the model configuration is not found
   */
  constructor(nickname: string) {
    super(nickname);
    logger.debug(
      `Initializing AnthropicChatModelProvider with nickname: ${nickname}`,
    );
    const config = storage().models.get<IAnthropicChatModelConfig>(nickname);
    if (!config) {
      throw new Error(`Model configuration not found for ${nickname}`);
    }
    this.config = config;
    logger.info(`AnthropicChatModelProvider initialized for ${nickname}`);
  }

  /**
   * Configures a new Anthropic model
   * @param {string} nickname - The nickname for the new model configuration
   * @throws {Error} If configuration process is cancelled or fails
   */
  static readonly configure = async (nickname: string): Promise<void> => {
    logger.info(`Configuring Anthropic model with nickname: ${nickname}`);

    const config = storage().models.get<IAnthropicChatModelConfig>(nickname);

    // Prompt user for API key
    let apiKey = await vscode.window.showInputBox({
      ignoreFocusOut: true,
      value: config?.apiKey ?? "",
      valueSelection: [0, 0],
      placeHolder: "e.g., sk-ant-api03-qojTF59pEBxB7DZ...", // cspell:disable-line
      prompt: DEFAULT_HELP_PROMPT,
      title: "Flexpilot: Enter your Anthropic API key",
    });
    if (apiKey === undefined) {
      throw new Error("User cancelled API key input");
    }
    apiKey = apiKey.trim();

    // Prompt user for base URL
    const defaultBaseUrl = "https://api.anthropic.com/v1";
    let baseUrl = await vscode.window.showInputBox({
      ignoreFocusOut: true,
      value: config?.baseUrl ?? defaultBaseUrl,
      valueSelection: [0, 0],
      placeHolder: `e.g., ${defaultBaseUrl}`,
      prompt: DEFAULT_HELP_PROMPT,
      title: "Flexpilot: Enter the base URL for Anthropic API",
    });
    if (baseUrl === undefined) {
      throw new Error("User cancelled base URL input");
    }
    baseUrl = baseUrl.trim();

    // Prompt user to enter model name
    const defaultModel = "claude-3-5-sonnet-20240620";
    let model = await vscode.window.showInputBox({
      ignoreFocusOut: true,
      value: config?.model ?? "",
      valueSelection: [0, 0],
      placeHolder: `e.g., ${defaultModel}`,
      prompt: "Enter the model name",
      title: "Flexpilot: Enter the Anthropic API model name",
    });
    if (model === undefined) {
      throw new Error("User cancelled model input");
    }
    model = model.trim();

    // Test the connection credentials
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Flexpilot",
        cancellable: false,
      },
      async (progress) => {
        progress.report({
          message: "Testing connection credentials",
        });
        const anthropic = createAnthropic({
          apiKey: apiKey,
          baseURL: baseUrl,
        });
        logger.debug("Testing connection with a simple prompt");
        await generateText({
          prompt: "Hello",
          maxTokens: 3,
          model: anthropic.languageModel(model),
        });
        logger.info("Connection credentials test successful");
      },
    );

    // Save the selected model configuration
    logger.debug(`Saving model configuration for ${nickname}`);
    await storage().models.set<IAnthropicChatModelConfig>(nickname, {
      baseUrl: baseUrl,
      apiKey: apiKey,
      model: model,
      nickname: nickname,
      providerId: AnthropicChatModelProvider.providerId,
    });

    logger.info(`Model configuration saved for ${nickname}`);
  };

  /**
   * Creates and returns a LanguageModelV1 instance
   * @returns {Promise<LanguageModelV1>} A promise that resolves to a LanguageModelV1 instance
   */
  async model(): Promise<LanguageModelV1> {
    logger.debug(`Creating LanguageModelV1 instance for ${this.config.model}`);
    const anthropic = createAnthropic({
      baseURL: this.config.baseUrl,
      apiKey: this.config.apiKey,
    });
    logger.info(`LanguageModelV1 instance created for ${this.config.model}`);
    return anthropic.languageModel(this.config.model);
  }
}
