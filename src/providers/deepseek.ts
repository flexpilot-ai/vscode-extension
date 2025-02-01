import { Tokenizer } from "@flexpilot-ai/tokenizers";
import OpenAI from "openai";
import * as vscode from "vscode";
import {
  ICompletionModelConfig,
  ICompletionModelInvokeOptions,
  ICompletionModelProvider,
} from "../interfaces";
import { logger } from "../logger";
import { storage } from "../storage";
import { Tokenizers } from "../tokenizers";
import { getCompletionModelMetadata } from "../utilities";

/**
 * Configuration interface for DeepSeek Completion Model.
 */
export interface IDeepSeekCompletionModelConfig extends ICompletionModelConfig {
  baseUrl: string;
  apiKey: string;
}

/**
 * Default help prompt for DeepSeek configuration.
 */
const DEFAULT_HELP_PROMPT =
  "Click [here](https://docs.flexpilot.ai/model-providers/deepseek.html) for more information";

/**
 * Prompts the user to input their DeepSeek API key.
 * @param {string} [apiKey] - The current API key, if any.
 * @returns {Promise<string>} A promise that resolves to the input API key.
 * @throws {Error} If the user cancels the input.
 */
const getApiKeyInput = async (apiKey?: string): Promise<string> => {
  logger.debug("Prompting user for DeepSeek API key");
  const newApiKey = await vscode.window.showInputBox({
    title: "Flexpilot: Enter your DeepSeek API key",
    ignoreFocusOut: true,
    value: apiKey ?? "",
    validateInput: (value) =>
      !value?.trim() ? "API key cannot be empty" : undefined,
    valueSelection: [0, 0],
    placeHolder: "e.g., sk-mFzPtn4QYHOSJ...", // cspell:disable-line
    prompt: DEFAULT_HELP_PROMPT,
  });
  if (newApiKey === undefined) {
    throw new Error("User cancelled DeepSeek API key input");
  }
  logger.debug("DeepSeek API key input received");
  return newApiKey.trim();
};

/**
 * Provides completion model functionality for DeepSeek.
 * This provider implements the completion model interface for DeepSeek's API,
 * which is compatible with OpenAI's API format.
 */
export class DeepSeekCompletionModelProvider extends ICompletionModelProvider {
  static readonly providerName = "DeepSeek";
  static readonly providerId = "deepseek-completion";
  static readonly providerType = "completion" as const;
  private tokenizer!: Tokenizer;
  public readonly config: IDeepSeekCompletionModelConfig;

  constructor(nickname: string) {
    super(nickname);
    logger.debug(
      `Initializing DeepSeekCompletionModelProvider with nickname: ${nickname}`,
    );
    const config = storage.models.get<IDeepSeekCompletionModelConfig>(nickname);
    if (!config) {
      throw new Error(`Model configuration not found for ${nickname}`);
    }
    this.config = config;
    logger.info(`DeepSeekCompletionModelProvider initialized for ${nickname}`);
  }

  /**
   * Initializes the DeepSeek model provider.
   * @returns {Promise<void>} A promise that resolves when the provider is initialized.
   */
  async initialize(): Promise<void> {
    this.tokenizer = await Tokenizers.get(this.config.model);
  }

  readonly encode = async (text: string): Promise<number[]> => {
    logger.debug(`Encoding text: ${text.substring(0, 50)}...`);
    return this.tokenizer.encode(text, false);
  };

  readonly decode = async (tokens: number[]): Promise<string> => {
    logger.debug(`Decoding ${tokens.length} tokens`);
    return this.tokenizer.decode(tokens, false);
  };

  /**
   * Invokes the DeepSeek model with the given options.
   * @param {ICompletionModelInvokeOptions} options - The options for invoking the model.
   * @returns {Promise<string>} A promise that resolves to the model's response.
   * @throws {Error} If the model response is invalid or empty.
   */
  async invoke(options: ICompletionModelInvokeOptions): Promise<string> {
    logger.debug(`Invoking DeepSeek model: ${this.config.model} with options`);

    const openai = new OpenAI({
      apiKey: this.config.apiKey,
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

    const output = response.choices?.[0]?.text ?? "";
    logger.debug(`Model output: ${output.substring(0, 50)}...`);
    return output;
  }

  /**
   * Configures a new DeepSeek model
   * @param {string} nickname - The nickname for the new model configuration
   * @returns {Promise<void>} A promise that resolves when the configuration is complete
   * @throws {Error} If the configuration process fails
   */
  static readonly configure = async (nickname: string): Promise<void> => {
    logger.info(`Configuring DeepSeek model with nickname: ${nickname}`);

    // Load existing configuration
    const config = storage.models.get<IDeepSeekCompletionModelConfig>(nickname);

    // Prompt user for DeepSeek API key
    const apiKey = await getApiKeyInput(config?.apiKey);

    // Prompt user for DeepSeek base URL
    const defaultBaseUrl = "https://api.deepseek.com/beta";
    let baseUrl = await vscode.window.showInputBox({
      ignoreFocusOut: true,
      value: config?.baseUrl ?? defaultBaseUrl,
      valueSelection: [0, 0],
      placeHolder: `e.g., ${defaultBaseUrl}`,
      prompt: DEFAULT_HELP_PROMPT,
      title: "Flexpilot: Enter the base URL for DeepSeek API",
    });
    if (baseUrl === undefined) {
      throw new Error("User cancelled base URL input");
    }
    baseUrl = baseUrl.trim();

    // DeepSeek model will always be "deepseek-chat"
    const model = "deepseek-chat";

    // Download the selected model's tokenizer
    logger.info(`Downloading tokenizer for model: ${model}`);
    await Tokenizers.download(model);

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
        logger.debug("Testing connection credentials");
        const openai = new OpenAI({
          apiKey: apiKey,
          baseURL: baseUrl,
        });
        await openai.completions.create({
          model: model,
          max_tokens: 3,
          prompt: "How",
          suffix: "are you?",
        });
        logger.info("Connection credentials test successful");
      },
    );
    // Get metadata for the selected model
    const metadata = getCompletionModelMetadata(model);
    if (!metadata) {
      throw new Error("Unable to find model metadata");
    }

    // Save the model configuration
    logger.info(`Saving model configuration for: ${nickname}`);
    await storage.models.set<IDeepSeekCompletionModelConfig>(nickname, {
      contextWindow: metadata.contextWindow,
      baseUrl: baseUrl,
      apiKey: apiKey,
      model: model,
      nickname: nickname,
      providerId: DeepSeekCompletionModelProvider.providerId,
    });

    logger.info(`Successfully configured DeepSeek model: ${nickname}`);
  };
}
