import { createOpenAI } from "@ai-sdk/openai";
import { generateText, LanguageModelV1 } from "ai";
import OpenAI from "openai";
import * as vscode from "vscode";
import { IChatModelProvider, IModelConfig } from "../interfaces";
import { logger } from "../logger";
import { storage } from "../storage";

/**
 * Configuration interface for Generic Chat Model.
 */
interface IGenericChatModelConfig extends IModelConfig {
  apiKey: string;
  baseUrl: string;
}

/**
 * Default help prompt for Generic configuration.
 */
const DEFAULT_HELP_PROMPT =
  "Click [here](https://docs.flexpilot.ai/model-providers/generic.html) for more information";

/**
 * Generic Chat Model Provider class
 */
export class GenericChatModelProvider extends IChatModelProvider {
  static readonly providerName = "OpenAI Compatible";
  static readonly providerId = "generic-chat";
  static readonly providerType = "chat" as const;
  public readonly config: IGenericChatModelConfig;

  /**
   * Constructor for GenericChatModelProvider
   * @param {string} nickname - The nickname for the model configuration
   * @throws {Error} If the model configuration is not found
   */
  constructor(nickname: string) {
    super(nickname);
    logger.debug(
      `Initializing GenericChatModelProvider with nickname: ${nickname}`,
    );
    const config = storage().models.get<IGenericChatModelConfig>(nickname);
    if (!config) {
      throw new Error(`Model configuration not found for ${nickname}`);
    }
    this.config = config;
    logger.info(`GenericChatModelProvider initialized for ${nickname}`);
  }

  /**
   * Configures a new Generic chat model
   * @param {string} nickname - The nickname for the new model configuration
   * @returns {Promise<void>} A promise that resolves when the configuration is complete
   * @throws {Error} If the configuration process fails
   */
  static readonly configure = async (nickname: string): Promise<void> => {
    logger.info(`Configuring Generic chat model with nickname: ${nickname}`);

    const config = storage().models.get<IGenericChatModelConfig>(nickname);

    // Prompt user for base URL
    logger.debug("Prompting user for base URL");

    const baseUrl = await vscode.window.showInputBox({
      ignoreFocusOut: true,
      value: config?.baseUrl ?? "",
      validateInput: (value) =>
        !value?.trim() ? "Base URL cannot be empty" : undefined,
      valueSelection: [0, 0],
      placeHolder: `e.g., http://localhost:11434/v1`,
      prompt: DEFAULT_HELP_PROMPT,
      title: "Flexpilot: Enter your base URL",
    });

    if (baseUrl === undefined) {
      throw new Error("User cancelled base URL input");
    }
    logger.debug("base URL input received");

    // Prompt user for API key
    logger.debug("Prompting user for API key");

    const apiKey = await vscode.window.showInputBox({
      title: "Flexpilot: Enter your API key",
      ignoreFocusOut: true,
      value: config?.apiKey ?? "",
      validateInput: (value) =>
        !value?.trim() ? "API key cannot be empty" : undefined,
      valueSelection: [0, 0],
      placeHolder: "e.g., ollama", // cspell:disable-line
      prompt: DEFAULT_HELP_PROMPT,
    });

    if (apiKey === undefined) {
      throw new Error("User cancelled API key input");
    }
    logger.debug("API key input received");

    // Declare variables for model selection
    let modelsList: string[] = [];
    let model: string | undefined;

    // Auto Fetch available models
    try {
      const openai = new OpenAI({ apiKey: apiKey, baseURL: baseUrl });
      const models = await openai.models.list();
      logger.debug(`Fetched ${models.data.length} models`);
      modelsList = models.data.map((model) => model.id);
    } catch (error) {
      // Log error and continue with manual model entry
      logger.error(error as Error);
    }

    if (modelsList.length) {
      // Prompt user to select model ID
      model = await vscode.window.showQuickPick(modelsList, {
        placeHolder: "Select a chat model",
        ignoreFocusOut: true,
        title: "Flexpilot: Select the chat model",
      });
    } else {
      // Prompt user to manually enter model ID
      model = await vscode.window.showInputBox({
        title: "Flexpilot: Enter your model name",
        ignoreFocusOut: true,
        value: config?.model ?? "",
        validateInput: (value) =>
          !value?.trim() ? "Model name cannot be empty" : undefined,
        valueSelection: [0, 0],
        placeHolder: "e.g., llama3.2:1b",
        prompt: DEFAULT_HELP_PROMPT,
      });
    }

    // Check if user cancelled model selection
    if (model === undefined) {
      throw new Error("User cancelled model selection");
    }

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
        const openai = createOpenAI({
          compatibility: "compatible",
          baseURL: baseUrl,
          apiKey: apiKey,
        });
        logger.debug("Testing connection credentials");
        await generateText({
          prompt: "Hello",
          maxTokens: 3,
          model: openai.chat(model),
        });
        logger.info("Connection credentials test successful");
      },
    );

    // Save the model configuration
    logger.info(`Saving model configuration for: ${nickname}`);
    await storage().models.set<IGenericChatModelConfig>(nickname, {
      baseUrl: baseUrl,
      apiKey: apiKey,
      model: model,
      nickname: nickname,
      providerId: GenericChatModelProvider.providerId,
    });

    logger.info(`Successfully configured Generic chat model: ${nickname}`);
  };

  /**
   * Creates and returns a LanguageModelV1 instance
   * @returns {Promise<LanguageModelV1>} A promise that resolves to a LanguageModelV1 instance
   */
  async model(): Promise<LanguageModelV1> {
    logger.debug(`Creating LanguageModelV1 instance for ${this.config.model}`);
    const openai = createOpenAI({
      compatibility: "compatible",
      baseURL: this.config.baseUrl,
      apiKey: this.config.apiKey,
    });
    const model = openai.chat(this.config.model);
    logger.debug(`LanguageModelV1 instance created for ${this.config.model}`);
    return model;
  }
}
