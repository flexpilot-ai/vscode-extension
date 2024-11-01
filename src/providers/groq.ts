import { createOpenAI } from "@ai-sdk/openai";
import { generateText, LanguageModelV1 } from "ai";
import OpenAI from "openai";
import * as vscode from "vscode";
import { IChatModelProvider, IModelConfig } from "../interfaces";
import { logger } from "../logger";
import { storage } from "../storage";

/**
 * Extended interface for GroqCloud chat model configuration
 */
interface IGroqCloudChatModelConfig extends IModelConfig {
  apiKey: string;
  baseUrl: string;
}

/**
 * Interface representing a GroqCloud model
 * @see https://console.groqcloud.com/docs/api-reference#models
 */
interface IGroqCloudModel {
  id: string;
  active: boolean;
}

/**
 * Default help prompt for GroqCloud API key configuration
 */
const DEFAULT_HELP_PROMPT =
  "Click [here](https://docs.flexpilot.ai/model-providers/groq.html) for more information";

/**
 * GroqCloud Chat Model Provider class
 * Manages the configuration and creation of GroqCloud chat models
 */
export class GroqCloudChatModelProvider extends IChatModelProvider {
  static readonly providerName = "GroqCloud";
  static readonly providerId = "groqcloud-chat";
  static readonly providerType = "chat" as const;
  public readonly config: IGroqCloudChatModelConfig;

  /**
   * Constructor for GroqCloudChatModelProvider
   * @param {string} nickname - The nickname for the model configuration
   * @throws {Error} If the model configuration is not found
   */
  constructor(nickname: string) {
    super(nickname);
    logger.debug(
      `Initializing GroqCloudChatModelProvider with nickname: ${nickname}`,
    );
    const config = storage().models.get<IGroqCloudChatModelConfig>(nickname);
    if (!config) {
      throw new Error(`Model configuration not found for ${nickname}`);
    }
    this.config = config;
    logger.info(`GroqCloudChatModelProvider initialized for ${nickname}`);
  }

  /**
   * Configures a new GroqCloud model
   * @param {string} nickname - The nickname for the new model configuration
   * @throws {Error} If configuration process is cancelled or fails
   */
  static readonly configure = async (nickname: string): Promise<void> => {
    logger.info(`Configuring GroqCloud model with nickname: ${nickname}`);

    const config = storage().models.get<IGroqCloudChatModelConfig>(nickname);

    // Prompt user for API key
    let apiKey = await vscode.window.showInputBox({
      ignoreFocusOut: true,
      value: config?.apiKey ?? "",
      valueSelection: [0, 0],
      placeHolder: "e.g., gsk_vgAcnlKOXdklg2AWLUv...", // cspell:disable-line
      prompt: DEFAULT_HELP_PROMPT,
      title: "Flexpilot: Enter your GroqCloud API key",
    });
    if (apiKey === undefined) {
      throw new Error("User cancelled API key input");
    }
    apiKey = apiKey.trim();

    // Prompt user for base URL
    const defaultBaseUrl = "https://api.groq.com/openai/v1";
    let baseUrl = await vscode.window.showInputBox({
      ignoreFocusOut: true,
      value: config?.baseUrl ?? defaultBaseUrl,
      valueSelection: [0, 0],
      placeHolder: `e.g., ${defaultBaseUrl}`,
      prompt: DEFAULT_HELP_PROMPT,
      title: "Flexpilot: Enter the base URL for GroqCloud API",
    });
    if (baseUrl === undefined) {
      throw new Error("User cancelled base URL input");
    }
    baseUrl = baseUrl.trim();

    // Fetch available models from GroqCloud API
    const response = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Flexpilot",
        cancellable: true,
      },
      async (progress) => {
        progress.report({
          message: "Fetching available models",
        });
        const openai = new OpenAI({
          apiKey: apiKey,
          baseURL: baseUrl,
        });
        logger.debug("Fetching models from GroqCloud API");
        const models = await openai.models.list();
        logger.debug(`Fetched ${models.data.length} models`);
        return models.data as unknown as IGroqCloudModel[];
      },
    );

    // Filter out models that are not supported for content generation
    const modelPickUpItems: vscode.QuickPickItem[] = [];
    for (const model of response) {
      if (model.active) {
        modelPickUpItems.push({ label: model.id });
      }
    }

    // Prompt user to select a model
    const model = await vscode.window.showQuickPick(modelPickUpItems, {
      placeHolder: "Select a chat model",
      ignoreFocusOut: true,
      title: "Flexpilot: Select the GroqCloud model",
    });
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
        const groqcloud = createOpenAI({
          apiKey: apiKey,
          baseURL: baseUrl,
        });
        logger.debug("Testing connection with a simple prompt");
        await generateText({
          prompt: "Hello",
          maxTokens: 3,
          model: groqcloud.chat(model.label),
        });
        logger.info("Connection credentials test successful");
      },
    );

    // Save the selected model configuration
    logger.debug(`Saving model configuration for ${nickname}`);
    await storage().models.set<IGroqCloudChatModelConfig>(nickname, {
      baseUrl: baseUrl,
      apiKey: apiKey,
      model: model.label,
      nickname: nickname,
      providerId: GroqCloudChatModelProvider.providerId,
    });

    logger.info(`Model configuration saved for ${nickname}`);
  };

  /**
   * Creates and returns a LanguageModelV1 instance
   * @returns {Promise<LanguageModelV1>} A promise that resolves to a LanguageModelV1 instance
   */
  async model(): Promise<LanguageModelV1> {
    logger.debug(`Creating LanguageModelV1 instance for ${this.config.model}`);
    const groqcloud = createOpenAI({
      baseURL: this.config.baseUrl,
      apiKey: this.config.apiKey,
    });
    logger.info(`LanguageModelV1 instance created for ${this.config.model}`);
    return groqcloud.chat(this.config.model);
  }
}
