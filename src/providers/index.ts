import { LOCATIONS } from "../constants";
import { events } from "../events";
import { ILocationName, IModelType } from "../interfaces";
import { logger } from "../logger";
import { storage } from "../storage";
import { AnthropicChatModelProvider } from "./anthropic";
import {
  AzureOpenAIChatModelProvider,
  AzureOpenAICompletionModelProvider,
} from "./azure";
import { DeepSeekCompletionModelProvider } from "./deepseek";
import { GenericChatModelProvider } from "./generic";
import { GoogleChatModelProvider } from "./google";
import { GroqCloudChatModelProvider } from "./groq";
import { LlamaCppCompletionModelProvider } from "./llamacpp";
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
  LlamaCppCompletionModelProvider,
  DeepSeekCompletionModelProvider,
] as const;

/**
 * Interfaces for the model provider class.
 */
export type IModelProvider<T extends IModelType = IModelType> = InstanceType<
  Extract<(typeof ModelProviders)[number], { providerType: T }>
>;

export class Test123 {}

/**
 * Manages model providers for different locations.
 * Implements the Singleton pattern to ensure a single instance across the application.
 */
export class ModelProviderManager {
  private readonly modelProviders: Map<ILocationName, IModelProvider> =
    new Map();
  private static instance: ModelProviderManager;

  private constructor(extensionContext = storage.getContext()) {
    // Check for changes in the lastProviderUpdatedAt secret to re-initialize providers
    extensionContext.subscriptions.push(
      events.onFire(async (event) => {
        logger.debug(`Event Action Started: ${event.name}`);
        if (event.name === "modelProvidersUpdated") {
          logger.info("Re-initializing model providers");
          this.updateProviders();
        }
      }),
    );
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
   * Updates the provider for all locations based on the usage preference.
   */
  public async updateProviders(): Promise<void> {
    logger.info("Initializing providers for all locations");
    await Promise.all(
      LOCATIONS.map(async (location) => {
        try {
          logger.info(`Updating provider for location: ${location.name}`);

          // Trigger once before updating provider
          if (location.name === "Inline Completion") {
            events.fire({
              name: "inlineCompletionProviderUpdated",
              payload: { updatedAt: Date.now() },
            });
          }

          // Delete the existing provider
          this.modelProviders.delete(location.name);

          // Get usage preference for the location
          const usagePreference = storage.usage.get(location.name);

          if (!usagePreference) {
            return logger.info(
              `No usage preference found for location: ${location.name}`,
            );
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
          const modelProvider = new ModelProvider(usagePreference.nickname);
          await modelProvider.initialize();
          this.modelProviders.set(location.name, modelProvider);
          logger.info(
            `Provider updated successfully for location: ${location.name}`,
          );

          // Trigger once after updating provider
          if (location.name === "Inline Completion") {
            events.fire({
              name: "inlineCompletionProviderUpdated",
              payload: { updatedAt: Date.now() },
            });
          }
        } catch (error) {
          logger.error(error as Error);
          logger.notifyError("Error updating model provider configuration");
        }
      }),
    );
  }
}
