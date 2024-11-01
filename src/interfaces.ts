import { LanguageModelV1 } from "ai";
import * as vscode from "vscode";
import packageJson from "../package.json";
import { LOCATIONS, PROVIDER_TYPES } from "./constants";

/**
 * Type for package.json.
 */
export type IPackageJson = typeof packageJson & {
  contributes?: {
    [key: string]: unknown;
    menus: {
      [key: string]: unknown;
    };
  };
  [key: string]: unknown;
};

/**
 * Interface for completion model invoke options.
 */
export interface ICompletionModelInvokeOptions {
  maxTokens: number;
  stop: string[];
  temperature: number;
  signal: AbortSignal;
  messages: { prefix: string; suffix: string };
}

/**
 * Interface extending vscode.ChatResult with additional metadata.
 */
export interface IChatResult extends vscode.ChatResult {
  metadata: { response: string; request: string };
}

/**
 * Type for location names based on model type.
 */
export type ILocationName<T extends IModelType = IModelType> = Extract<
  (typeof LOCATIONS)[number],
  { type: T }
>["name"];

/**
 * Type for model types.
 */
export type IModelType = (typeof PROVIDER_TYPES)[number];

/**
 * Interface for usage preferences.
 */
export interface IUsagePreference {
  providerId: string;
  nickname: string;
  locationName: ILocationName;
}

/**
 * Abstract base class for model providers.
 */
abstract class IModelProviderBase {
  static readonly providerName: string;
  static readonly providerId: string;
  static readonly providerType: IModelType;

  constructor(public nickname: string) {}

  abstract config: IModelConfig;

  static configure(nickname: string): Promise<void> {
    throw new Error(`Method not implemented for ${nickname}`);
  }
}

/**
 * Abstract class for chat model providers.
 */
export abstract class IChatModelProvider extends IModelProviderBase {
  static readonly providerType: "chat";

  abstract model(): Promise<LanguageModelV1>;
}

/**
 * Abstract class for completion model providers.
 */
export abstract class ICompletionModelProvider extends IModelProviderBase {
  static readonly providerType: "completion";

  abstract encode: (text: string) => Promise<number[]>;

  abstract decode: (tokens: number[]) => Promise<string>;

  abstract invoke(options: ICompletionModelInvokeOptions): Promise<string>;
}

/**
 * Interface for model configuration.
 */
export interface IModelConfig {
  nickname: string;
  model: string;
  providerId: string;
}

/**
 * Interface for chat model invoke options.
 */
export interface ICompletionModelConfig extends IModelConfig {
  contextWindow: number;
}

/**
 * Interface representing the configuration for a language.
 */
export interface ILanguageConfig {
  comment: {
    start: string;
    end: string;
  };
  markdown: string;
}
