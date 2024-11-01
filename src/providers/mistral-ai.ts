import { createMistral } from "@ai-sdk/mistral";
import { Tokenizer } from "@flexpilot-ai/tokenizers";
import { Mistral } from "@mistralai/mistralai";
import { generateText, LanguageModelV1 } from "ai";
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
 * Configuration interface for Mistral AI Completion Model.
 */
interface IMistralAICompletionModelConfig extends ICompletionModelConfig {
  apiKey: string;
  endpoint: string;
}

/**
 * Configuration interface for Mistral AI Chat Model.
 */
interface IMistralAIChatModelConfig extends IModelConfig {
  apiKey: string;
  endpoint: string;
}

/**
 * Default help prompt for Mistral AI configuration.
 */
const DEFAULT_HELP_PROMPT =
  "Click [here](https://docs.flexpilot.ai/model-providers/mistral-ai.html) for more information";

/**
 * Prompts the user to input their Mistral AI API key.
 * @param {string} [apiKey] - The current API key, if any.
 * @returns {Promise<string>} A promise that resolves to the input API key.
 * @throws {Error} If the user cancels the input.
 */
const getApiKeyInput = async (apiKey?: string): Promise<string> => {
  logger.debug("Prompting user for Mistral AI API key");
  const newApiKey = await vscode.window.showInputBox({
    title: "Flexpilot: Enter your Mistral AI API key",
    ignoreFocusOut: true,
    value: apiKey ?? "",
    validateInput: (value) =>
      !value?.trim() ? "API key cannot be empty" : undefined,
    valueSelection: [0, 0],
    placeHolder: "e.g., 8tAkaS8oamghqYPXqQXRcRFOquwyoWES", // cspell:disable-line
    prompt: DEFAULT_HELP_PROMPT,
  });
  if (newApiKey === undefined) {
    throw new Error("User cancelled Mistral AI API key input");
  }
  logger.debug("Mistral AI API key input received");
  return newApiKey.trim();
};

/**
 * Prompts the user to input their Mistral AI endpoint.
 * @param {string} [endpoint] - The current endpoint, if any.
 * @returns {Promise<string>} A promise that resolves to the input endpoint.
 * @throws {Error} If the user cancels the input.
 */
const getEndpointInput = async (endpoint?: string): Promise<string> => {
  logger.debug("Prompting user for Mistral AI endpoint");
  const defaultEndpoint = "https://codestral.mistral.ai";
  const newEndpoint = await vscode.window.showInputBox({
    ignoreFocusOut: true,
    value: endpoint ?? defaultEndpoint,
    validateInput: (value) =>
      !value?.trim() ? "Endpoint cannot be empty" : undefined,
    valueSelection: [0, 0],
    placeHolder: `e.g., ${defaultEndpoint}`,
    prompt: DEFAULT_HELP_PROMPT,
    title: "Flexpilot: Enter your Mistral AI endpoint",
  });
  if (newEndpoint === undefined) {
    throw new Error("User cancelled Mistral AI endpoint input");
  }
  logger.debug("Mistral AI endpoint input received");
  return newEndpoint.trim();
};

/**
 * Provides completion model functionality for Mistral AI.
 */
export class MistralAICompletionModelProvider extends ICompletionModelProvider {
  static readonly providerId = "mistral-completion";
  static readonly providerName = "Mistral AI";
  static readonly providerType = "completion" as const;
  private readonly tokenizer: Tokenizer;
  public readonly config: IMistralAICompletionModelConfig;

  /**
   * Constructs a new MistralAICompletionModelProvider.
   * @param {string} nickname - The nickname for the model configuration.
   * @throws {Error} If the model configuration is not found.
   */
  constructor(nickname: string) {
    super(nickname);
    logger.info(
      `Initializing MistralAICompletionModelProvider with nickname: ${nickname}`,
    );
    const config =
      storage().models.get<IMistralAICompletionModelConfig>(nickname);
    if (!config) {
      throw new Error(`Model configuration not found for ${nickname}`);
    }
    this.config = config;
    this.tokenizer = Tokenizers.get(this.config.model);
    logger.debug(
      `MistralAICompletionModelProvider initialized for ${nickname}`,
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
   * Invokes the Mistral AI completion model.
   * @param {ICompletionModelInvokeOptions} options - The options for the invocation.
   * @returns {Promise<string>} A promise that resolves to the model's output.
   */
  readonly invoke = async (
    options: ICompletionModelInvokeOptions,
  ): Promise<string> => {
    logger.info("Invoking Mistral AI completion model");
    const mistral = new Mistral({
      serverURL: this.config.endpoint,
      apiKey: this.config.apiKey,
      retryConfig: { strategy: "none" },
    });
    const response = await mistral.fim.complete(
      {
        model: this.config.model,
        prompt: options.messages.prefix,
        maxTokens: options.maxTokens,
        temperature: options.temperature,
        stop: options.stop,
        suffix: options.messages.suffix,
      },
      { fetchOptions: { signal: options.signal } },
    );
    let output = response.choices?.[0]?.message?.content ?? "";
    if (options.messages.prefix.endsWith(" ")) {
      output = output.trimStart();
    }
    if (options.messages.suffix.startsWith(" ")) {
      output = output.trimEnd();
    }
    logger.debug(`Model output: ${output.substring(0, 50)}...`);
    return output;
  };

  /**
   * Configures the Mistral AI completion model.
   * @param {string} nickname - The nickname for the model configuration.
   * @returns {Promise<void>}
   */
  static readonly configure = async (nickname: string): Promise<void> => {
    logger.info(
      `Configuring Mistral AI completion model with nickname: ${nickname}`,
    );
    const config =
      storage().models.get<IMistralAICompletionModelConfig>(nickname);

    // Prompt user for endpoint
    const endpoint = await getEndpointInput(config?.endpoint);

    // Prompt user for API key
    const apiKey = await getApiKeyInput(config?.apiKey);

    // Prompt user for model name
    const defaultModelName = "codestral-latest";
    let model = await vscode.window.showInputBox({
      ignoreFocusOut: true,
      value: config?.model ?? defaultModelName,
      valueSelection: [0, 0],
      validateInput: (value) =>
        !getCompletionModelMetadata(value) ? "Invalid model name" : undefined,
      placeHolder: `e.g., ${defaultModelName}`,
      prompt: DEFAULT_HELP_PROMPT,
      title: "Flexpilot: Enter the model's name",
    });
    if (model === undefined) {
      throw new Error("User cancelled model name input");
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
        const mistral = new Mistral({
          serverURL: endpoint,
          apiKey: apiKey,
          retryConfig: { strategy: "none" },
        });
        await mistral.fim.complete({
          prompt: "How",
          model: model,
          maxTokens: 3,
          suffix: "are you?",
        });
        logger.info("Connection credentials test successful");
      },
    );

    // Download the selected model's tokenizer
    await Tokenizers.download(model);

    // Get metadata for the selected model
    const metadata = getCompletionModelMetadata(model);
    if (!metadata) {
      throw new Error("Unable to find model metadata");
    }

    // Save the model configuration
    await storage().models.set<IMistralAICompletionModelConfig>(nickname, {
      endpoint: endpoint,
      apiKey: apiKey,
      model: model,
      nickname: nickname,
      contextWindow: metadata.contextWindow,
      providerId: MistralAICompletionModelProvider.providerId,
    });
    logger.info(
      `Mistral AI completion model configuration saved for ${nickname}`,
    );
  };
}

/**
 * Provides chat model functionality for Mistral AI.
 */
export class MistralAIChatModelProvider extends IChatModelProvider {
  static readonly providerId = "mistral-chat";
  static readonly providerName = "Mistral AI";
  static readonly providerType = "chat" as const;
  public readonly config: IMistralAIChatModelConfig;

  /**
   * Constructs a new MistralAIChatModelProvider.
   * @param {string} nickname - The nickname for the model configuration.
   * @throws {Error} If the model configuration is not found.
   */
  constructor(nickname: string) {
    super(nickname);
    logger.info(
      `Initializing MistralAIChatModelProvider with nickname: ${nickname}`,
    );
    const config = storage().models.get<IMistralAIChatModelConfig>(nickname);
    if (!config) {
      throw new Error(`Model configuration not found for ${nickname}`);
    }
    this.config = config;
    logger.debug(`MistralAIChatModelProvider initialized for ${nickname}`);
  }

  /**
   * Configures the Mistral AI chat model.
   * @param {string} nickname - The nickname for the model configuration.
   * @returns {Promise<void>}
   */
  static readonly configure = async (nickname: string): Promise<void> => {
    logger.info(`Configuring Mistral AI chat model with nickname: ${nickname}`);
    const config = storage().models.get<IMistralAIChatModelConfig>(nickname);

    // Prompt user for endpoint
    const endpoint = await getEndpointInput(config?.endpoint);

    // Prompt user for API key
    const apiKey = await getApiKeyInput(config?.apiKey);

    // Prompt user for model name
    const defaultModelName = "codestral-latest";
    let model = await vscode.window.showInputBox({
      ignoreFocusOut: true,
      value: config?.model ?? defaultModelName,
      valueSelection: [0, 0],
      placeHolder: `e.g., ${defaultModelName}`,
      prompt: DEFAULT_HELP_PROMPT,
      title: "Flexpilot: Enter the model's name",
    });
    if (model === undefined) {
      throw new Error("User cancelled model name input");
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
        const mistral = createMistral({
          apiKey: apiKey,
          baseURL: endpoint + "/v1",
        });
        await generateText({
          prompt: "Hello",
          maxTokens: 3,
          model: mistral.chat(model),
        });
        logger.info("Connection credentials test successful");
      },
    );

    // Save the model configuration
    await storage().models.set<IMistralAIChatModelConfig>(nickname, {
      endpoint: endpoint,
      apiKey: apiKey,
      model: model,
      nickname: nickname,
      providerId: MistralAIChatModelProvider.providerId,
    });
    logger.info(`Mistral AI chat model configuration saved for ${nickname}`);
  };

  /**
   * Creates and returns a LanguageModelV1 instance.
   * @returns {Promise<LanguageModelV1>} A promise that resolves to a LanguageModelV1 instance.
   */
  async model(): Promise<LanguageModelV1> {
    logger.debug(
      `Creating LanguageModelV1 instance for ${this.config.nickname}`,
    );
    const mistral = createMistral({
      apiKey: this.config.apiKey,
      baseURL: this.config.endpoint + "/v1",
    });
    return mistral.chat(this.config.model);
  }
}
