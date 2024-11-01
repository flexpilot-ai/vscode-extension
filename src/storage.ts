import * as vscode from "vscode";
import packageJson from "../package.json";
import { ILocationName, IModelConfig, IUsagePreference } from "./interfaces";
import { logger } from "./logger";

/**
 * IKeyValueStore interface to define the structure of permanent key-value store.
 */
interface IKeyValueStore {
  "github.support": boolean;
  "completions.disabled.languages": string[];
  "argv.path": string;
}

/**
 * IWorkspaceConfigKeys type to define the keys of workspace configuration.
 */
type IWorkspaceConfigKeys =
  keyof (typeof packageJson)["contributes"]["configuration"]["properties"];

/**
 * StorageManager class provides a centralized state management mechanism for the extension.
 * It implements the Singleton pattern to ensure a single instance of the state across the application.
 */
export class StorageManager {
  public readonly context: vscode.ExtensionContext;
  private static instance: StorageManager;
  private githubSession: vscode.AuthenticationSession | undefined;

  /**
   * Creates a new instance of StorageManager.
   * @param {vscode.ExtensionContext} extensionContext - The extension context provided by VS Code
   */
  private constructor(extensionContext: vscode.ExtensionContext) {
    this.context = extensionContext;
    logger.info("StorageManager instance created");
  }

  /**
   * Creates a new instance of StorageManager.
   * @param {vscode.ExtensionContext} extensionContext - The extension context provided by VS Code
   */
  public static createInstance(extensionContext: vscode.ExtensionContext) {
    if (!StorageManager.instance) {
      StorageManager.instance = new StorageManager(extensionContext);
    }
  }

  /**
   * Returns the singleton instance of StorageManager.
   */
  public static getInstance(): StorageManager {
    if (!StorageManager.instance) {
      throw logger.notifyError("storage accessed before activation");
    }
    return StorageManager.instance;
  }

  /**
   * Retrieves a value from permanent storage.
   */
  public get = <T1 extends keyof IKeyValueStore>(
    key: T1,
  ): IKeyValueStore[T1] | undefined => {
    logger.debug(`Getting global state for key: ${key}`);
    return this.context.globalState.get(key);
  };

  /**
   * Sets a value in permanent storage.
   */
  public set = async <T1 extends keyof IKeyValueStore>(
    key: T1,
    value: IKeyValueStore[T1] | undefined,
  ): Promise<void> => {
    logger.debug(`Setting global state for key: ${key}`);
    return this.context.globalState.update(key, value);
  };

  public session = {
    /**
     * Retrieves the GitHub session from the storage.
     */
    get: (): vscode.AuthenticationSession => {
      if (!this.githubSession) {
        throw new Error("GitHub session not found");
      }
      return this.githubSession;
    },

    /**
     * Sets the GitHub session.
     */
    set: (session: vscode.AuthenticationSession | undefined) => {
      this.githubSession = session;
    },
  };

  public models = {
    /**
     * Retrieves a value from model providers.
     */
    get: <T extends IModelConfig>(key: string): T | undefined => {
      const storageKey = `model.providers.${key}`;
      logger.debug(`Getting model config for key: ${storageKey}`);
      return this.context.globalState.get<T>(storageKey);
    },

    /**
     * Sets a value in model providers.
     */
    set: async <T extends IModelConfig>(key: string, value?: T) => {
      const storageKey = `model.providers.${key}`;
      logger.debug(`Setting model config for key: ${storageKey}`);
      return this.context.globalState.update(storageKey, value);
    },

    /**
     * Lists all the model providers.
     */
    list: (): string[] => {
      logger.debug(`Listing global state for key: model.providers`);
      return this.context.globalState
        .keys()
        .filter((item) => item.startsWith("model.providers."))
        .map((item) => item.replace("model.providers.", ""));
    },
  };

  public usage = {
    /**
     * Retrieves a value from usage preferences.
     */
    get: (key: ILocationName): IUsagePreference | undefined => {
      const storageKey = `usage.preferences.${key}`;
      logger.debug(`Getting usage preference for key: ${storageKey}`);
      return this.context.globalState.get<IUsagePreference>(storageKey);
    },

    /**
     * Sets a value in usage preferences.
     */
    set: async <T extends IUsagePreference>(
      key: ILocationName,
      value: T | undefined,
    ) => {
      const storageKey = `usage.preferences.${key}`;
      logger.debug(`Setting usage preference for key: ${storageKey}`);
      return this.context.globalState.update(storageKey, value);
    },
  };

  public workspace = {
    /**
     * Retrieves a value from workspace configuration.
     */
    get: <T>(key: IWorkspaceConfigKeys): T => {
      logger.debug(`Getting workspace config for key: flexpilot.${key}`);
      return vscode.workspace.getConfiguration().get<T>(key) as T;
    },

    /**
     * Sets a value in workspace configuration.
     */
    set: async <T>(key: IWorkspaceConfigKeys, value: T): Promise<void> => {
      logger.debug(`Setting workspace config for key: flexpilot.${key}`);
      return vscode.workspace
        .getConfiguration()
        .update(key, value, vscode.ConfigurationTarget.Global);
    },
  };
}

// Export the storage manager instance
export const storage = () => StorageManager.getInstance();
