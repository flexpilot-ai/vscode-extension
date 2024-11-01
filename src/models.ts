import * as vscode from "vscode";

/**
 * ProxyModelProvider class provides functionality for the proxy model.
 * It implements the Singleton pattern to ensure a single instance across the application.
 */
export class ProxyModelProvider {
  private static instance: ProxyModelProvider | null = null;

  /**
   * Private constructor to prevent direct instantiation.
   * Initializes the proxy model provider.
   */
  private constructor() {
    // Register the proxy model
    vscode.lm.registerChatModelProvider(
      "proxy",
      {
        provideLanguageModelResponse() {
          throw new Error("Method not implemented.");
        },
        provideTokenCount() {
          throw new Error("Method not implemented.");
        },
      },
      {
        maxInputTokens: 10000,
        family: "proxy",
        name: "proxy",
        isDefault: true,
        vendor: "flexpilot",
        version: "1.0.0",
        maxOutputTokens: 10000,
      },
    );
  }

  /**
   * Returns the singleton instance of ProxyModelProvider.
   * @returns {ProxyModelProvider} The singleton instance.
   */
  public static register(): ProxyModelProvider {
    if (!ProxyModelProvider.instance) {
      ProxyModelProvider.instance = new ProxyModelProvider();
    }
    return ProxyModelProvider.instance;
  }
}
