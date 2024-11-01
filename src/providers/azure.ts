import { createAzure } from "@ai-sdk/azure";
import { Tokenizer } from "@flexpilot-ai/tokenizers";
import { generateText, LanguageModelV1 } from "ai";
import { AzureOpenAI } from "openai";
import * as vscode from "vscode";
import {
  IChatModelProvider,
  ICompletionModelConfig,
  ICompletionModelInvokeOptions,
  ICompletionModelProvider,
  IModelConfig,
} from "../interfaces";
import { logger } from "../logger";
import { storage } from "../storage";
import { Tokenizers } from "../tokenizers";
import { getCompletionModelMetadata } from "../utilities";

/**
 * Configuration interface for Azure OpenAI Chat Model.
 */
interface AzureOpenAIChatModelConfig extends IModelConfig {
  apiKey: string;
  baseUrl: string;
}

/**
 * Configuration interface for Azure OpenAI Completion Model.
 */
interface IAzureOpenAICompletionModelConfig extends ICompletionModelConfig {
  apiKey: string;
  baseUrl: string;
}

/**
 * Default help prompt for Azure OpenAI configuration.
 */
const DEFAULT_HELP_PROMPT =
  "Click [here](https://docs.flexpilot.ai/model-providers/azure-openai.html) for more information";

/**
 * Prompts the user to input their Azure OpenAI API key.
 * @param {string} [apiKey] - The current API key, if any.
 * @returns {Promise<string>} A promise that resolves to the input API key.
 * @throws {Error} If the user cancels the input.
 */
const getApiKeyInput = async (apiKey?: string): Promise<string> => {
  logger.debug("Prompting user for Azure OpenAI API key");
  const newApiKey = await vscode.window.showInputBox({
    title: "Flexpilot: Enter your Azure OpenAI API key",
    ignoreFocusOut: true,
    value: apiKey ?? "",
    validateInput: (value) =>
      !value?.trim() ? "API key cannot be empty" : undefined,
    valueSelection: [0, 0],
    placeHolder: "e.g., upydshyx1rlleghhe1zw4hri4z80pvdn", // cspell:disable-line
    prompt: DEFAULT_HELP_PROMPT,
  });
  if (newApiKey === undefined) {
    throw new Error("User cancelled Azure OpenAI API key input");
  }
  logger.debug("Azure OpenAI API key input received");
  return newApiKey.trim();
};

/**
 * Prompts the user to input their Azure OpenAI base URL.
 * @param {string} [baseUrl] - The current base URL, if any.
 * @returns {Promise<string>} A promise that resolves to the input base URL.
 * @throws {Error} If the user cancels the input.
 */
const getBaseUrlInput = async (baseUrl?: string): Promise<string> => {
  logger.debug("Prompting user for Azure OpenAI base URL");
  const newBaseUrl = await vscode.window.showInputBox({
    placeHolder:
      "e.g., https://{resourceName}.openai.azure.com/openai/deployments/{deploymentId}",
    ignoreFocusOut: true,
    value: baseUrl ?? "",
    validateInput: (value) =>
      !value?.trim() ? "Base URL cannot be empty" : undefined,
    valueSelection: [0, 0],
    prompt: DEFAULT_HELP_PROMPT,
    title: "Flexpilot: Enter your Azure OpenAI base URL",
  });
  if (newBaseUrl === undefined) {
    throw new Error("User cancelled Azure OpenAI base URL input");
  }
  logger.debug("Azure OpenAI base URL input received");
  return newBaseUrl.trim();
};

/**
 * Azure OpenAI Completion Model Provider class
 */
export class AzureOpenAICompletionModelProvider extends ICompletionModelProvider {
  static readonly providerName = "Azure OpenAI";
  static readonly providerId = "azure-openai-completion";
  static readonly providerType = "completion" as const;
  private readonly tokenizer: Tokenizer;
  public readonly config: IAzureOpenAICompletionModelConfig;

  /**
   * Constructor for AzureOpenAICompletionModelProvider
   * @param {string} nickname - The nickname for the model configuration
   * @throws {Error} If the model configuration is not found
   */
  constructor(nickname: string) {
    super(nickname);
    logger.info(
      `Initializing AzureOpenAICompletionModelProvider with nickname: ${nickname}`,
    );
    const config =
      storage().models.get<IAzureOpenAICompletionModelConfig>(nickname);
    if (!config) {
      throw new Error(`Model configuration not found for ${nickname}`);
    }
    this.config = config;
    this.tokenizer = Tokenizers.get(this.config.model);
    logger.debug(
      `AzureOpenAICompletionModelProvider initialized for ${nickname}`,
    );
  }

  /**
   * Encodes the given text into tokens.
   * @param {string} text - The text to encode.
   * @returns {Promise<number[]>} A promise that resolves to an array of token ids.
   */
  readonly encode = async (text: string): Promise<number[]> => {
    logger.debug(`Encoding text: ${text.substring(0, 50)}...`);
    return this.tokenizer.encode(text, false);
  };

  /**
   * Decodes the given tokens into text.
   * @param {number[]} tokens - The tokens to decode.
   * @returns {Promise<string>} A promise that resolves to the decoded text.
   */
  readonly decode = async (tokens: number[]): Promise<string> => {
    logger.debug(`Decoding ${tokens.length} tokens`);
    return this.tokenizer.decode(tokens, false);
  };

  /**
   * Configures a new Azure OpenAI model
   * @param {string} nickname - The nickname for the new model configuration
   * @returns {Promise<void>} A promise that resolves when the configuration is complete
   * @throws {Error} If the configuration process fails
   */
  static readonly configure = async (nickname: string): Promise<void> => {
    logger.info(`Configuring Azure OpenAI model with nickname: ${nickname}`);

    // Load existing configuration
    const config =
      storage().models.get<IAzureOpenAICompletionModelConfig>(nickname);

    // Prompt user for Azure OpenAI API key
    const apiKey = await getApiKeyInput(config?.apiKey);

    // Prompt user for Azure OpenAI base URL
    const baseUrl = await getBaseUrlInput(config?.baseUrl);

    // Test the connection credentials
    let modelId!: string;
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
        logger.debug("Testing connection credentials");
        const openai = new AzureOpenAI({
          apiKey: apiKey,
          apiVersion: "2024-06-01",
          baseURL: baseUrl,
        });
        const response = await openai.completions.create({
          prompt: "How",
          suffix: "are you?",
          max_tokens: 3,
          model: baseUrl.split("/").pop() ?? "",
        });
        modelId = response.model;
        logger.info("Connection credentials test successful");
      },
    );

    // Get the model metadata
    const metadata = getCompletionModelMetadata(modelId);
    if (!metadata) {
      throw new Error(`Model metadata not found for: ${modelId}`);
    }

    // Download the selected model's tokenizer
    logger.info(`Downloading tokenizer for model: ${modelId}`);
    await Tokenizers.download(modelId);

    // Save the model configuration
    logger.info(`Saving model configuration for: ${nickname}`);
    await storage().models.set<IAzureOpenAICompletionModelConfig>(nickname, {
      contextWindow: metadata.contextWindow,
      baseUrl: baseUrl,
      apiKey: apiKey,
      model: modelId,
      nickname: nickname,
      providerId: AzureOpenAICompletionModelProvider.providerId,
    });

    logger.info(`Successfully configured Azure OpenAI model: ${nickname}`);
  };

  /**
   * Invokes the Azure OpenAI model with the given options.
   * @param {ICompletionModelInvokeOptions} options - The options for invoking the model.
   * @returns {Promise<string>} A promise that resolves to the model's response.
   * @throws {Error} If the invocation fails.
   */
  async invoke(options: ICompletionModelInvokeOptions): Promise<string> {
    logger.info(`Invoking Azure OpenAI model: ${this.config.model}`);
    logger.debug("Generating text with Azure OpenAI model");

    const openai = new AzureOpenAI({
      apiKey: this.config.apiKey,
      apiVersion: "2024-06-01",
      baseURL: this.config.baseUrl,
    });
    const response = await openai.completions.create(
      {
        prompt: options.messages.prefix,
        model: this.config.model,
        max_tokens: options.maxTokens,
        stop: options.stop,
        suffix: options.messages.suffix,
        temperature: options.temperature,
      },
      { signal: options.signal },
    );
    logger.debug(
      `Model output: ${response.choices[0].text.substring(0, 50)}...`,
    );
    return response.choices[0].text;
  }
}

/**
 * Azure OpenAI Chat Model Provider class
 */
export class AzureOpenAIChatModelProvider extends IChatModelProvider {
  static readonly providerName = "Azure OpenAI";
  static readonly providerId = "azure-openai-chat";
  static readonly providerType = "chat" as const;
  public readonly config: AzureOpenAIChatModelConfig;

  /**
   * Constructor for AzureOpenAIChatModelProvider
   * @param {string} nickname - The nickname for the model configuration
   * @throws {Error} If the model configuration is not found
   */
  constructor(nickname: string) {
    super(nickname);
    logger.debug(
      `Initializing AzureOpenAIChatModelProvider with nickname: ${nickname}`,
    );
    const config = storage().models.get<AzureOpenAIChatModelConfig>(nickname);
    if (!config) {
      throw new Error(`Model configuration not found for ${nickname}`);
    }
    this.config = config;
    logger.info(`AzureOpenAIChatModelProvider initialized for ${nickname}`);
  }

  /**
   * Configures a new Azure OpenAI chat model
   * @param {string} nickname - The nickname for the new model configuration
   * @returns {Promise<void>} A promise that resolves when the configuration is complete
   * @throws {Error} If the configuration process fails
   */
  static readonly configure = async (nickname: string): Promise<void> => {
    logger.info(
      `Configuring Azure OpenAI chat model with nickname: ${nickname}`,
    );

    const config = storage().models.get<AzureOpenAIChatModelConfig>(nickname);

    // Prompt user for Azure OpenAI API key
    const apiKey = await getApiKeyInput(config?.apiKey);

    // Prompt user for Azure OpenAI base URL
    const baseUrl = await getBaseUrlInput(config?.baseUrl);

    // Test the connection credentials
    let modelId!: string;
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
        const openai = createAzure({
          apiKey: apiKey,
          baseURL: baseUrl,
        });
        logger.debug("Testing connection credentials");
        const response = await generateText({
          prompt: "Hello",
          maxTokens: 3,
          model: openai.chat(""),
        });
        modelId = response.response.modelId;
        logger.info("Connection credentials test successful");
      },
    );

    // Save the model configuration
    logger.info(`Saving model configuration for: ${nickname}`);
    await storage().models.set<AzureOpenAIChatModelConfig>(nickname, {
      baseUrl: baseUrl,
      apiKey: apiKey,
      model: modelId,
      nickname: nickname,
      providerId: AzureOpenAIChatModelProvider.providerId,
    });

    logger.info(`Successfully configured Azure OpenAI chat model: ${nickname}`);
  };

  /**
   * Creates and returns a LanguageModelV1 instance
   * @returns {Promise<LanguageModelV1>} A promise that resolves to a LanguageModelV1 instance
   * @throws {Error} If the model creation fails
   */
  async model(): Promise<LanguageModelV1> {
    logger.debug(`Creating LanguageModelV1 instance for ${this.config.model}`);

    const openai = createAzure({
      baseURL: this.config.baseUrl,
      apiKey: this.config.apiKey,
    });
    const model = openai.chat("");
    logger.debug(`LanguageModelV1 instance created for ${this.config.model}`);
    return model;
  }
}
