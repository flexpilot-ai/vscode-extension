import * as vscode from "vscode";
import {
  ICompletionModelConfig,
  ICompletionModelInvokeOptions,
  ICompletionModelProvider,
} from "../interfaces";
import { logger } from "../logger";
import { storage } from "../storage";

/**
 * Configuration interface for llama.cpp Completion Model.
 */
interface ILlamaCppCompletionModelConfig extends ICompletionModelConfig {
  baseUrl: string;
}

/**
 * Default help prompt for llama.cpp configuration.
 */
const DEFAULT_HELP_PROMPT =
  "Click [here](https://docs.flexpilot.ai/model-providers/llamacpp.html) for more information";

/**
 * Response interface for tokenization requests.
 * Contains the tokenized representation of input text.
 */
interface LlamaCppTokenizeResponse {
  tokens: number[];
}

/**
 * Response interface for detokenization requests.
 * Contains the text decoded from input tokens.
 */
interface LlamaCppDetokenizeResponse {
  content: string;
}

/**
 * Response interface for model infill requests.
 * Contains the generated text and optional metadata about the generation process.
 */
interface LlamaCppInfillResponse {
  content?: string;
  tokens_cached?: number;
  truncated?: boolean;
  timings?: {
    prompt_n?: number;
    prompt_ms?: number;
    prompt_per_second?: number;
    predicted_n?: number;
    predicted_ms?: number;
    predicted_per_second?: number;
  };
}

/**
 * Provides completion model functionality for llama.cpp.
 * This provider implements the completion model interface for llama.cpp's HTTP server,
 * allowing integration with locally running llama.cpp instances.
 */
export class LlamaCppCompletionModelProvider extends ICompletionModelProvider {
  static readonly providerName = "llama.cpp";
  static readonly providerId = "llama.cpp-completion";
  static readonly providerType = "completion" as const;
  public readonly config: ILlamaCppCompletionModelConfig;

  constructor(nickname: string) {
    super(nickname);
    logger.debug(
      `Initializing llama.cpp Completion Model Provider with nickname: ${nickname}`,
    );
    const config = storage.models.get<ILlamaCppCompletionModelConfig>(nickname);
    if (!config) {
      throw new Error(`Model configuration not found for ${nickname}`);
    }
    this.config = config;
    logger.info(
      `llama.cpp Completion Model Provider initialized for ${nickname}`,
    );
  }

  readonly encode = async (text: string): Promise<number[]> => {
    logger.debug(`Encoding text: ${text.substring(0, 50)}...`);
    const response = await fetch(`${this.config.baseUrl}/tokenize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: text, add_special: false }),
    });
    if (!response.ok) {
      throw new Error(
        `Failed to tokenize text: ${response.status} ${response.statusText}`,
      );
    }
    const result = (await response.json()) as LlamaCppTokenizeResponse;
    return result.tokens;
  };

  readonly decode = async (tokens: number[]): Promise<string> => {
    logger.debug(`Decoding ${tokens.length} tokens`);
    const response = await fetch(`${this.config.baseUrl}/detokenize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tokens: tokens }),
    });
    if (!response.ok) {
      throw new Error(
        `Failed to detokenize tokens: ${response.status} ${response.statusText}`,
      );
    }
    const result = (await response.json()) as LlamaCppDetokenizeResponse;
    return result.content;
  };

  async initialize(): Promise<void> {
    logger.info("Initializing llama.cpp");
  }

  static readonly configure = async (nickname: string): Promise<void> => {
    logger.info(`Configuring llama.cpp model with nickname: ${nickname}`);
    // Load existing configuration
    const config = storage.models.get<ILlamaCppCompletionModelConfig>(nickname);

    const defaultBaseUrl = "http://localhost:8012";
    let baseUrl = await vscode.window.showInputBox({
      ignoreFocusOut: true,
      value: config?.baseUrl ?? defaultBaseUrl,
      valueSelection: [0, 0],
      placeHolder: `e.g., ${defaultBaseUrl}`,
      prompt: DEFAULT_HELP_PROMPT,
      title: "Flexpilot: Enter the base URL for llama.cpp API",
    });
    if (baseUrl === undefined) {
      throw new Error("User cancelled base URL input");
    }
    baseUrl = baseUrl.trim();

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
        const response = await fetch(`${baseUrl}/completion`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt: "Hello",
            n_predict: 3,
            stream: false,
          }),
        });
        if (!response.ok) {
          throw new Error(`Connection test failed: ${response.statusText}`);
        }
        logger.info("Connection credentials test successful");
      },
    );

    logger.info(`Saving model configuration for: ${nickname}`);
    storage.models.set<ILlamaCppCompletionModelConfig>(nickname, {
      contextWindow: 32768, // Default context window, can be adjusted based on model
      baseUrl: baseUrl,
      model: "", // llama.cpp server will run with given model
      nickname: nickname,
      providerId: LlamaCppCompletionModelProvider.providerId,
    });
    logger.info(`Successfully configured llama.cpp: ${nickname}`);
  };

  async invoke(options: ICompletionModelInvokeOptions): Promise<string> {
    logger.debug("Generating text with llama.cpp");

    const response = await fetch(`${this.config.baseUrl}/infill`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        input_prefix: options.messages?.prefix ?? "",
        input_suffix: options.messages?.suffix ?? "",
        n_predict: options.maxTokens ?? 100,
        stop: options.stop ?? [],
        stream: false,
      }),
      signal: options.signal,
    });

    if (!response.ok) {
      const error = await response.text().catch(() => "Unknown error");
      throw new Error(
        `Failed to invoke llama.cpp model: ${response.status} ${response.statusText}\n${error}`,
      );
    }

    const result = (await response.json()) as LlamaCppInfillResponse;

    if (result.truncated) {
      logger.warn("Response was truncated due to context length");
    }

    if (result.timings) {
      logger.debug(
        `Generation stats: ${result.timings.predicted_n} tokens in ${
          result.timings.predicted_ms
        }ms (${result.timings.predicted_per_second?.toFixed(2)} tokens/s)`,
      );
    }

    const output = result.content ?? "";
    logger.debug(`Model output: ${output.substring(0, 50)}...`);
    return output;
  }
}
