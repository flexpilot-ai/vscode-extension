import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { generateText, LanguageModelV1 } from "ai";
import axios from "axios";
import * as vscode from "vscode";
import { IChatModelProvider, IModelConfig } from "../interfaces";
import { logger } from "../logger";
import { storage } from "../storage";

/**
 * Extended interface for Google Generative AI chat model configuration
 */
interface IGoogleGenerativeAIChatModelConfig extends IModelConfig {
  apiKey: string;
  baseUrl: string;
}

/**
 * Interface representing a Google AI model
 * @see https://ai.google.dev/api/models#Model
 */
interface IGoogleModel {
  name: string;
  supportedGenerationMethods: string[];
}

const DEFAULT_HELP_PROMPT =
  "Click [here](https://docs.flexpilot.ai/model-providers/google.html) for more information";

/**
 * Google Generative AI Chat Model Provider class
 */
export class GoogleChatModelProvider extends IChatModelProvider {
  static readonly providerName = "Google Generative AI";
  static readonly providerId = "google-generative-ai-chat";
  static readonly providerType = "chat" as const;
  public readonly config: IGoogleGenerativeAIChatModelConfig;

  /**
   * Constructor for GoogleChatModelProvider
   * @param {string} nickname - The nickname for the model configuration
   * @throws {Error} If the model configuration is not found
   */
  constructor(nickname: string) {
    super(nickname);
    logger.debug(
      `Initializing GoogleChatModelProvider with nickname: ${nickname}`,
    );
    const config =
      storage().models.get<IGoogleGenerativeAIChatModelConfig>(nickname);
    if (!config) {
      throw new Error(`Model configuration not found for ${nickname}`);
    }
    this.config = config;
    logger.info(`GoogleChatModelProvider initialized for ${nickname}`);
  }

  /**
   * Configures a new Google Generative AI model
   * @param {string} nickname - The nickname for the new model configuration
   * @throws {Error} If configuration process is cancelled or fails
   */
  static readonly configure = async (nickname: string): Promise<void> => {
    logger.info(
      `Configuring Google Generative AI model with nickname: ${nickname}`,
    );

    const config =
      storage().models.get<IGoogleGenerativeAIChatModelConfig>(nickname);

    // Prompt user for API key
    let apiKey = await vscode.window.showInputBox({
      ignoreFocusOut: true,
      value: config?.apiKey ?? "",
      valueSelection: [0, 0],
      placeHolder: "e.g., A123SyBMH486BbQe684JHG2ASZ2-RKmmVe-X11M", // cspell:disable-line
      prompt: DEFAULT_HELP_PROMPT,
      title: "Flexpilot: Enter your Google Generative AI API key",
    });
    if (apiKey === undefined) {
      throw new Error("User cancelled API key input");
    }
    apiKey = apiKey.trim();

    // Prompt user for base URL
    const defaultBaseUrl = "https://generativelanguage.googleapis.com/v1beta";
    let baseUrl = await vscode.window.showInputBox({
      ignoreFocusOut: true,
      value: config?.baseUrl ?? defaultBaseUrl,
      valueSelection: [0, 0],
      placeHolder: `e.g., ${defaultBaseUrl}`,
      prompt: DEFAULT_HELP_PROMPT,
      title: "Flexpilot: Enter the base URL for Google Generative AI",
    });
    if (baseUrl === undefined) {
      throw new Error("User cancelled base URL input");
    }
    baseUrl = baseUrl.trim();

    // Fetch available models from Google Generative AI API
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
        return await axios.get(`${baseUrl}/models?key=${apiKey}`);
      },
    );

    // Filter out models that are not supported for content generation
    const modelPickUpItems: vscode.QuickPickItem[] = [];
    for (const model of response.data.models as IGoogleModel[]) {
      if (
        model.name.split("-")[0].split("/")[1] === "gemini" &&
        parseFloat(model.name.split("-")[1]) > 1 &&
        model.supportedGenerationMethods.includes("generateContent")
      ) {
        modelPickUpItems.push({ label: model.name });
      }
    }

    // Prompt user to select a model
    const model = await vscode.window.showQuickPick(modelPickUpItems, {
      placeHolder: "Select a chat model",
      ignoreFocusOut: true,
      title: "Flexpilot: Select the Google Generative AI model",
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
        const google = createGoogleGenerativeAI({
          apiKey: apiKey,
          baseURL: baseUrl,
        });
        await generateText({
          prompt: "Hello",
          maxTokens: 3,
          model: google.chat(model.label),
        });
        logger.info("Connection credentials test successful");
      },
    );

    // Save the selected model configuration
    await storage().models.set<IGoogleGenerativeAIChatModelConfig>(nickname, {
      baseUrl: baseUrl,
      apiKey: apiKey,
      model: model.label,
      nickname: nickname,
      providerId: GoogleChatModelProvider.providerId,
    });

    logger.info(`Model configuration saved for ${nickname}`);
  };

  /**
   * Creates and returns a LanguageModelV1 instance
   * @returns {Promise<LanguageModelV1>} A promise that resolves to a LanguageModelV1 instance
   */
  async model(): Promise<LanguageModelV1> {
    logger.debug(`Creating LanguageModelV1 instance for ${this.config.model}`);
    const google = createGoogleGenerativeAI({
      baseURL: this.config.baseUrl,
      apiKey: this.config.apiKey,
    });
    return google.chat(this.config.model);
  }
}
