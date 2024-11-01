import { createOpenAI } from "@ai-sdk/openai";
import { Tokenizer } from "@flexpilot-ai/tokenizers";
import { generateText, LanguageModelV1 } from "ai";
import OpenAI from "openai";
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
 * Configuration interface for OpenAI Chat Model.
 */
interface IOpenAIChatModelConfig extends IModelConfig {
  project?: string;
  apiKey: string;
  baseUrl: string;
  organization?: string;
}

/**
 * Configuration interface for OpenAI Completion Model.
 */
interface IOpenAICompletionModelConfig extends ICompletionModelConfig {
  project?: string;
  apiKey: string;
  baseUrl: string;
  organization?: string;
}

/**
 * Default help prompt for OpenAI configuration.
 */
const DEFAULT_HELP_PROMPT =
  "Click [here](https://docs.flexpilot.ai/model-providers/openai.html) for more information";

/**
 * Prompts the user to input their OpenAI API key.
 * @param {string} [apiKey] - The current API key, if any.
 * @returns {Promise<string>} A promise that resolves to the input API key.
 * @throws {Error} If the user cancels the input.
 */
const getApiKeyInput = async (apiKey?: string): Promise<string> => {
  logger.debug("Prompting user for OpenAI API key");
  const newApiKey = await vscode.window.showInputBox({
    title: "Flexpilot: Enter your OpenAI API key",
    ignoreFocusOut: true,
    value: apiKey ?? "",
    validateInput: (value) =>
      !value?.trim() ? "API key cannot be empty" : undefined,
    valueSelection: [0, 0],
    placeHolder: "e.g., sk-proj-mFzPtn4QYHOSJ...", // cspell:disable-line
    prompt: DEFAULT_HELP_PROMPT,
  });
  if (newApiKey === undefined) {
    throw new Error("User cancelled OpenAI API key input");
  }
  logger.debug("OpenAI API key input received");
  return newApiKey.trim();
};

/**
 * Prompts the user to input their OpenAI base URL.
 * @param {string} [baseUrl] - The current base URL, if any.
 * @returns {Promise<string>} A promise that resolves to the input base URL.
 * @throws {Error} If the user cancels the input.
 */
const getBaseUrlInput = async (baseUrl?: string): Promise<string> => {
  logger.debug("Prompting user for OpenAI base URL");
  const defaultBaseUrl = "https://api.openai.com/v1";
  const newBaseUrl = await vscode.window.showInputBox({
    ignoreFocusOut: true,
    value: baseUrl ?? defaultBaseUrl,
    validateInput: (value) =>
      !value?.trim() ? "Base URL cannot be empty" : undefined,
    valueSelection: [0, 0],
    placeHolder: `e.g., ${defaultBaseUrl}`,
    prompt: DEFAULT_HELP_PROMPT,
    title: "Flexpilot: Enter your OpenAI base URL",
  });
  if (newBaseUrl === undefined) {
    throw new Error("User cancelled OpenAI base URL input");
  }
  logger.debug("OpenAI base URL input received");
  return newBaseUrl.trim();
};

/**
 * Prompts the user to input their OpenAI organization.
 * @param {string} [organization] - The current organization, if any.
 * @returns {Promise<string | undefined>} A promise that resolves to the input organization or undefined.
 * @throws {Error} If the user cancels the input.
 */
const getOrganizationInput = async (
  organization?: string,
): Promise<string | undefined> => {
  logger.debug("Prompting user for OpenAI organization");
  const newOrganization = await vscode.window.showInputBox({
    title: "Flexpilot: Enter your OpenAI organization",
    ignoreFocusOut: true,
    value: organization ?? "",
    valueSelection: [0, 0],
    placeHolder: "(default is empty) e.g., org-lcXqJgdBUtlBkgkNgtORSuNc", // cspell:disable-line
    prompt: DEFAULT_HELP_PROMPT,
  });
  if (newOrganization === undefined) {
    throw new Error("User cancelled OpenAI organization input");
  }
  logger.debug("OpenAI organization input received");
  return newOrganization.trim() || undefined;
};

/**
 * Prompts the user to input their OpenAI project.
 * @param {string} [project] - The current project, if any.
 * @returns {Promise<string | undefined>} A promise that resolves to the input project or undefined.
 * @throws {Error} If the user cancels the input.
 */
const getProjectInput = async (
  project?: string,
): Promise<string | undefined> => {
  logger.debug("Prompting user for OpenAI project");
  const newProject = await vscode.window.showInputBox({
    title: "Flexpilot: Enter your OpenAI project",
    ignoreFocusOut: true,
    value: project ?? "",
    valueSelection: [0, 0],
    placeHolder: "(default is empty) e.g., proj_codfdIGEeiojIZoGdWwhJoXc", // cspell:disable-line
    prompt: DEFAULT_HELP_PROMPT,
  });
  if (newProject === undefined) {
    throw new Error("User cancelled OpenAI project input");
  }
  logger.debug("OpenAI project input received");
  return newProject.trim() || undefined;
};

/**
 * OpenAI Completion Model Provider class
 */
export class OpenAICompletionModelProvider extends ICompletionModelProvider {
  static readonly providerName = "OpenAI";
  static readonly providerId = "openai-completion";
  static readonly providerType = "completion" as const;
  private readonly tokenizer: Tokenizer;
  public readonly config: IOpenAICompletionModelConfig;

  /**
   * Constructor for OpenAICompletionModelProvider
   * @param {string} nickname - The nickname for the model configuration
   * @throws {Error} If the model configuration is not found
   */
  constructor(nickname: string) {
    super(nickname);
    logger.info(
      `Initializing OpenAICompletionModelProvider with nickname: ${nickname}`,
    );
    const config = storage().models.get<IOpenAICompletionModelConfig>(nickname);
    if (!config) {
      throw new Error(`Model configuration not found for ${nickname}`);
    }
    this.config = config;
    this.tokenizer = Tokenizers.get(this.config.model);
    logger.debug(`OpenAICompletionModelProvider initialized for ${nickname}`);
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
   * Configures a new OpenAI model
   * @param {string} nickname - The nickname for the new model configuration
   * @returns {Promise<void>} A promise that resolves when the configuration is complete
   * @throws {Error} If the configuration process fails
   */
  static readonly configure = async (nickname: string): Promise<void> => {
    logger.info(`Configuring OpenAI model with nickname: ${nickname}`);

    // Load existing configuration
    const config = storage().models.get<IOpenAICompletionModelConfig>(nickname);

    // Prompt user for OpenAI API key
    const apiKey = await getApiKeyInput(config?.apiKey);

    // Prompt user for OpenAI base URL
    const baseUrl = await getBaseUrlInput(config?.baseUrl);

    // Prompt user for OpenAI organization
    const organization = await getOrganizationInput(config?.organization);

    // Prompt user for OpenAI project
    const project = await getProjectInput(config?.project);

    // Fetch available models from OpenAI API
    const modelsList = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Flexpilot",
        cancellable: true,
      },
      async (progress) => {
        progress.report({ message: "Fetching available models" });
        const openai = new OpenAI({
          apiKey: apiKey,
          baseURL: baseUrl,
          organization: organization,
          project: project,
        });
        logger.debug("Fetching models from OpenAI API");
        const models = await openai.models.list();
        logger.debug(`Fetched ${models.data.length} models`);
        return models.data;
      },
    );

    // Prepare model pick-up items
    const modelPickUpItems: vscode.QuickPickItem[] = [];
    for (const model of modelsList) {
      logger.debug(`Checking model configuration for: ${model.id}`);
      const metadata = getCompletionModelMetadata(model.id);
      if (metadata) {
        modelPickUpItems.push({ label: model.id });
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
        const openai = new OpenAI({
          apiKey: apiKey,
          baseURL: baseUrl,
          organization: organization,
          project: project,
        });
        await openai.completions.create({
          model: model.label,
          max_tokens: 3,
          prompt: "How",
          suffix: "are you?",
        });
        logger.info("Connection credentials test successful");
      },
    );
    // Get metadata for the selected model
    const metadata = getCompletionModelMetadata(model.label);
    if (!metadata) {
      throw new Error("Unable to find model metadata");
    }

    // Save the model configuration
    logger.info(`Saving model configuration for: ${nickname}`);
    await storage().models.set<IOpenAICompletionModelConfig>(nickname, {
      contextWindow: metadata.contextWindow,
      organization: organization,
      project: project,
      baseUrl: baseUrl,
      apiKey: apiKey,
      model: model.label,
      nickname: nickname,
      providerId: OpenAICompletionModelProvider.providerId,
    });

    logger.info(`Successfully configured OpenAI model: ${nickname}`);
  };

  /**
   * Invokes the OpenAI model with the given options.
   * @param {ICompletionModelInvokeOptions} options - The options for invoking the model.
   * @returns {Promise<string>} A promise that resolves to the model's response.
   */
  async invoke(options: ICompletionModelInvokeOptions): Promise<string> {
    logger.info(`Invoking OpenAI model: ${this.config.model}`);
    logger.debug("Generating text with OpenAI model");
    const openai = new OpenAI({
      project: this.config.project,
      apiKey: this.config.apiKey,
      baseURL: this.config.baseUrl,
      organization: this.config.organization,
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
 * OpenAI Chat Model Provider class
 */
export class OpenAIChatModelProvider extends IChatModelProvider {
  static readonly providerName = "OpenAI";
  static readonly providerId = "openai-chat";
  static readonly providerType = "chat" as const;
  public readonly config: IOpenAIChatModelConfig;

  /**
   * Constructor for OpenAIChatModelProvider
   * @param {string} nickname - The nickname for the model configuration
   * @throws {Error} If the model configuration is not found
   */
  constructor(nickname: string) {
    super(nickname);
    logger.debug(
      `Initializing OpenAIChatModelProvider with nickname: ${nickname}`,
    );
    const config = storage().models.get<IOpenAIChatModelConfig>(nickname);
    if (!config) {
      throw new Error(`Model configuration not found for ${nickname}`);
    }
    this.config = config;
    logger.info(`OpenAIChatModelProvider initialized for ${nickname}`);
  }

  /**
   * Configures a new OpenAI chat model
   * @param {string} nickname - The nickname for the new model configuration
   * @returns {Promise<void>} A promise that resolves when the configuration is complete
   * @throws {Error} If the configuration process fails
   */
  static readonly configure = async (nickname: string): Promise<void> => {
    logger.info(`Configuring OpenAI chat model with nickname: ${nickname}`);

    const config = storage().models.get<IOpenAIChatModelConfig>(nickname);

    // Prompt user for OpenAI API key
    const apiKey = await getApiKeyInput(config?.apiKey);

    // Prompt user for OpenAI base URL
    const baseUrl = await getBaseUrlInput(config?.baseUrl);

    // Prompt user for OpenAI organization
    const organization = await getOrganizationInput(config?.organization);

    // Prompt user for OpenAI project
    const project = await getProjectInput(config?.project);

    // Fetch available models from OpenAI API
    const modelsList = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Flexpilot",
        cancellable: true,
      },
      async (progress) => {
        progress.report({ message: "Fetching available models" });
        const openai = new OpenAI({
          apiKey: apiKey,
          baseURL: baseUrl,
          organization: organization,
          project: project,
        });
        logger.debug("Fetching models from OpenAI API");
        const models = await openai.models.list();
        logger.debug(`Fetched ${models.data.length} models`);
        return models.data;
      },
    );

    // Prepare model pick-up items
    const modelPickUpItems = {
      valid: [
        {
          kind: vscode.QuickPickItemKind.Separator,
          label: "Chat Models",
        },
      ] as vscode.QuickPickItem[],
      others: [
        {
          kind: vscode.QuickPickItemKind.Separator,
          label: "Other Models",
        },
      ] as vscode.QuickPickItem[],
    };
    for (const model of modelsList) {
      if (model.object !== "model") {
        continue;
      } else if (model.id.includes("instruct")) {
        modelPickUpItems.others.push({ label: model.id });
      } else if (model.id.includes("realtime")) {
        modelPickUpItems.others.push({ label: model.id });
      } else if (model.id.startsWith("gpt")) {
        modelPickUpItems.valid.push({ label: model.id });
      } else {
        modelPickUpItems.others.push({ label: model.id });
      }
    }

    const model = await vscode.window.showQuickPick(
      Object.values(modelPickUpItems).flat(),
      {
        placeHolder: "Select a chat model",
        ignoreFocusOut: true,
        title: "Flexpilot: Select the chat model",
      },
    );
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
          compatibility: "strict",
          baseURL: baseUrl,
          apiKey: apiKey,
          project: project,
          organization: organization,
        });
        logger.debug("Testing connection credentials");
        await generateText({
          prompt: "Hello",
          maxTokens: 3,
          model: openai.chat(model.label),
        });
        logger.info("Connection credentials test successful");
      },
    );

    // Save the model configuration
    logger.info(`Saving model configuration for: ${nickname}`);
    await storage().models.set<IOpenAIChatModelConfig>(nickname, {
      organization: organization,
      project: project,
      baseUrl: baseUrl,
      apiKey: apiKey,
      model: model.label,
      nickname: nickname,
      providerId: OpenAIChatModelProvider.providerId,
    });

    logger.info(`Successfully configured OpenAI chat model: ${nickname}`);
  };

  /**
   * Creates and returns a LanguageModelV1 instance
   * @returns {Promise<LanguageModelV1>} A promise that resolves to a LanguageModelV1 instance
   */
  async model(): Promise<LanguageModelV1> {
    logger.debug(`Creating LanguageModelV1 instance for ${this.config.model}`);
    const openai = createOpenAI({
      compatibility: "strict",
      baseURL: this.config.baseUrl,
      apiKey: this.config.apiKey,
      project: this.config.project,
      organization: this.config.organization,
    });
    const model = openai.chat(this.config.model);
    logger.debug(`LanguageModelV1 instance created for ${this.config.model}`);
    return model;
  }
}
