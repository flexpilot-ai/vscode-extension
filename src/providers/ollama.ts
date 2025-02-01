import { Tokenizer } from "@flexpilot-ai/tokenizers";
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
 * Configuration interface for Ollama Completion Model.
 */
interface IOllamaCompletionModelConfig extends ICompletionModelConfig {
  baseUrl: string;
}

/**
 * Default help prompt for Ollama configuration.
 */
const DEFAULT_HELP_PROMPT =
  "Click [here](https://docs.flexpilot.ai/model-providers/ollama.html) for more information";

/**
 * Response interface for Ollama model information.
 * Contains details about a specific model including its metadata and capabilities.
 */
interface OllamaModel {
  name: string;
  model: string;
  modified_at: string;
  size: number;
  digest: string;
  details: {
    format: string;
    family: string;
    families?: string[];
    parameter_size?: string;
    quantization_level?: string;
  };
}

/**
 * Response interface for Ollama model list endpoint.
 * Contains an array of available models on the Ollama server.
 */
interface OllamaTagsResponse {
  models: OllamaModel[];
}

/**
 * Response interface for Ollama text generation.
 * Contains the generated text and metadata about the generation process.
 */
interface OllamaGenerateResponse {
  model: string;
  response: string;
  done: boolean;
  context?: number[];
  created_at?: string;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_duration?: number;
  eval_duration?: number;
  eval_count?: number;
  prompt_eval_count?: number;
}

/**
 * Ollama Completion Model Provider class
 */
export class OllamaCompletionModelProvider extends ICompletionModelProvider {
  static readonly providerName = "Ollama";
  static readonly providerId = "ollama-completion";
  static readonly providerType = "completion" as const;
  private tokenizer!: Tokenizer;
  public readonly config: IOllamaCompletionModelConfig;

  constructor(nickname: string) {
    super(nickname);
    logger.info(
      `Initializing OllamaCompletionModelProvider with nickname: ${nickname}`,
    );
    const config = storage.models.get<IOllamaCompletionModelConfig>(nickname);
    if (!config) {
      throw new Error(`Model configuration not found for ${nickname}`);
    }
    this.config = config;
    logger.debug(`OllamaCompletionModelProvider initialized for ${nickname}`);
  }

  readonly encode = async (text: string): Promise<number[]> => {
    logger.debug(`Encoding text: ${text.substring(0, 50)}...`);
    return this.tokenizer.encode(text, false);
  };

  readonly decode = async (tokens: number[]): Promise<string> => {
    logger.debug(`Decoding ${tokens.length} tokens`);
    return this.tokenizer.decode(tokens, false);
  };

  static readonly configure = async (nickname: string): Promise<void> => {
    logger.info(`Configuring Ollama model with nickname: ${nickname}`);

    // Load existing configuration
    const config = storage.models.get<IOllamaCompletionModelConfig>(nickname);

    // Prompt user for Ollama base URL
    const defaultBaseUrl = "http://localhost:11434";
    let baseUrl = await vscode.window.showInputBox({
      ignoreFocusOut: true,
      value: config?.baseUrl ?? defaultBaseUrl,
      valueSelection: [0, 0],
      placeHolder: `e.g., ${defaultBaseUrl}`,
      prompt: DEFAULT_HELP_PROMPT,
      title: "Flexpilot: Enter the base URL for Ollama API",
    });
    if (baseUrl === undefined) {
      throw new Error("User cancelled base URL input");
    }
    baseUrl = baseUrl.trim();

    // Fetch available models from Ollama API
    const modelsList = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Flexpilot",
        cancellable: true,
      },
      async (progress) => {
        progress.report({ message: "Fetching available models" });
        const response = await fetch(`${baseUrl}/api/tags`, {});
        if (!response.ok) {
          throw new Error(`Failed to fetch models: ${response.statusText}`);
        }
        const data = (await response.json()) as OllamaTagsResponse;
        return data.models || [];
      },
    );

    // Prepare model pick-up items
    const modelPickUpItems: vscode.QuickPickItem[] = [];
    const contextWindowMap = new Map<string, number>();
    for (const model of modelsList) {
      logger.debug(`Checking model configuration for: ${model.name}`);
      const metadata = getCompletionModelMetadata(model.name);
      if (metadata) {
        modelPickUpItems.push({ label: model.name });
        if (metadata.contextWindow) {
          contextWindowMap.set(model.name, metadata.contextWindow);
        }
      }
    }

    // Check if models were found
    if (modelPickUpItems.length === 0) {
      throw new Error("No models found for the given configuration");
    }

    // Prompt user to select a model
    const model = await vscode.window.showQuickPick(modelPickUpItems, {
      placeHolder: "Select a completion model",
      ignoreFocusOut: true,
      canPickMany: false,
      title: "Flexpilot: Select the completion model",
    });
    if (!model) {
      throw new Error("User cancelled model selection");
    }

    // Download the selected model's tokenizer
    logger.info(`Downloading tokenizer for model: ${model.label}`);
    await Tokenizers.download(model.label);

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
        const response = await fetch(`${baseUrl}/api/generate`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: model.label,
            options: {
              num_predict: 3,
            },
            prompt: "Hello",
            stream: false,
          }),
        });
        if (!response.ok) {
          throw new Error(`Connection test failed: ${response.statusText}`);
        }
        logger.info("Connection credentials test successful");
      },
    );

    // Save the model configuration
    logger.info(`Saving model configuration for: ${nickname}`);
    await storage.models.set<IOllamaCompletionModelConfig>(nickname, {
      contextWindow: contextWindowMap.get(model.label) || 4096,
      baseUrl: baseUrl,
      model: model.label,
      nickname: nickname,
      providerId: OllamaCompletionModelProvider.providerId,
    });

    logger.info(`Successfully configured Ollama model: ${nickname}`);
  };

  /**
   * Initializes the provider by setting up the tokenizer for the configured model.
   * @returns {Promise<void>} A promise that resolves when initialization is complete.
   * @throws {Error} If the tokenizer cannot be initialized.
   */
  async initialize(): Promise<void> {
    logger.debug(`Initializing tokenizer for model: ${this.config.model}`);
    this.tokenizer = await Tokenizers.get(this.config.model);
    logger.info(`Tokenizer initialized for model: ${this.config.model}`);
  }

  /**
   * Invokes the Ollama model to generate text.
   * @param {ICompletionModelInvokeOptions} options - The options for text generation.
   * @returns {Promise<string>} The generated text.
   * @throws {Error} If the request fails or returns invalid data.
   */
  async invoke(options: ICompletionModelInvokeOptions): Promise<string> {
    logger.debug("Invoking Ollama completion model");
    const {
      messages: { prefix, suffix },
    } = options;

    const response = await fetch(`${this.config.baseUrl}/api/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: this.config.model,
        prompt: prefix,
        options: {
          num_predict: options.maxTokens,
        },
        suffix: suffix || undefined,
        stream: false,
      }),
      signal: options.signal,
    });

    if (!response.ok) {
      throw new Error(`Ollama API request failed: ${response.statusText}`);
    }

    const data = (await response.json()) as OllamaGenerateResponse;
    if (data.eval_duration !== undefined && data.eval_count !== undefined) {
      // Convert nanoseconds to milliseconds
      const durationMs = (data.eval_duration / 1_000_000).toFixed(2);
      logger.debug(
        `Generation stats: ${data.eval_count?.toString()} tokens in ${durationMs}ms`,
      );
    }
    logger.debug(`Response: ${data.response}`);
    return data.response;
  }
}
