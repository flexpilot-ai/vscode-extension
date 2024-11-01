import { LOCATIONS } from "../constants";
import { ILocationName, IModelType } from "../interfaces";
import { logger } from "../logger";
import { statusIcon } from "../status-icon";
import { storage } from "../storage";
import { AnthropicChatModelProvider } from "./anthropic";
import {
  AzureOpenAIChatModelProvider,
  AzureOpenAICompletionModelProvider,
} from "./azure";
import { GenericChatModelProvider } from "./generic";
import { GoogleChatModelProvider } from "./google";
import { GroqCloudChatModelProvider } from "./groq";
import {
  MistralAIChatModelProvider,
  MistralAICompletionModelProvider,
} from "./mistral-ai";
import {
  OpenAIChatModelProvider,
  OpenAICompletionModelProvider,
} from "./openai";

/**
 * Array of available model provider classes.
 */
export const ModelProviders = [
  OpenAIChatModelProvider,
  GoogleChatModelProvider,
  MistralAIChatModelProvider,
  AzureOpenAIChatModelProvider,
  AnthropicChatModelProvider,
  GroqCloudChatModelProvider,
  GenericChatModelProvider,
  OpenAICompletionModelProvider,
  MistralAICompletionModelProvider,
  AzureOpenAICompletionModelProvider,
] as const;

/**
 * Interfaces for the model provider class.
 */
export type IModelProvider<T extends IModelType = IModelType> = InstanceType<
  Extract<(typeof ModelProviders)[number], { providerType: T }>
>;

/**
 * Manages model providers for different locations.
 * Implements the Singleton pattern to ensure a single instance across the application.
 */
export class ModelProviderManager {
  private readonly modelProviders: Map<ILocationName, IModelProvider> =
    new Map();
  private static instance: ModelProviderManager;

  private constructor() {
    logger.info("ModelProviderManager instance created");
  }

  /**
   * Gets the singleton instance of ModelProviderManager.
   * @returns The ModelProviderManager instance
   */
  public static getInstance(): ModelProviderManager {
    if (!ModelProviderManager.instance) {
      ModelProviderManager.instance = new ModelProviderManager();
      logger.debug("New ModelProviderManager instance created");
    }
    return ModelProviderManager.instance;
  }

  /**
   * Initializes providers for all locations.
   */
  public async initProviders(): Promise<void> {
    logger.info("Initializing providers for all locations");
    await Promise.all(
      LOCATIONS.map((location) => this.updateProvider(location.name)),
    );
    logger.info("All providers initialized");
  }

  /**
   * Gets the provider for a specific location.
   * @param locationName - The name of the location
   * @returns The model provider for the specified location, or undefined if not found
   */
  public getProvider<T extends IModelType>(
    locationName: ILocationName<T>,
  ): IModelProvider<T> | undefined {
    const provider = this.modelProviders.get(locationName);
    if (!provider) {
      logger.info(`No provider found for location ${locationName}`);
      return undefined;
    }
    logger.debug(`Provider retrieved for location: ${locationName}`);
    return provider as IModelProvider<T>;
  }

  /**
   * Updates the provider for a specific location.
   * @param locationName - The name of the location to update
   */
  private async updateProvider(locationName: ILocationName): Promise<void> {
    try {
      logger.info(`Updating provider for location: ${locationName}`);

      // Delete the existing provider
      this.modelProviders.delete(locationName);

      // Get usage preference for the location
      const usagePreference = storage().usage.get(locationName);

      if (!usagePreference) {
        return logger.info(
          `No usage preference found for location: ${locationName}`,
        );
      }

      // Find the location in the LOCATIONS array
      const location = LOCATIONS.find((item) => item.name === locationName);
      if (!location) {
        return logger.info(`Location not found: ${locationName}`);
      }

      // Find the appropriate ModelProvider based on the usage preference
      const ModelProvider = ModelProviders.find(
        (item) => item.providerId === usagePreference.providerId,
      );
      if (!ModelProvider) {
        return logger.info(
          `No matching ModelProvider found for providerId: ${usagePreference.providerId}`,
        );
      }

      // Create and set the new provider
      this.modelProviders.set(
        locationName,
        new ModelProvider(usagePreference.nickname),
      );
      logger.info(
        `Provider updated successfully for location: ${locationName}`,
      );

      // Update the status bar icon
      if (locationName === "Inline Completion") {
        statusIcon().updateStatusBarIcon();
      }
    } catch (error) {
      logger.error(error as Error);
      logger.notifyError("Error updating model provider configuration");
    }
  }
}
