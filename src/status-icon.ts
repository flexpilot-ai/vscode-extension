import * as vscode from "vscode";
import { logger } from "./logger";
import { ModelProviderManager } from "./providers";
import { storage } from "./storage";

/**
 * Status icon manager to handle the status icon and state.
 * Created as a singleton to ensure a single instance across the application.
 */
class StatusIconManager extends vscode.Disposable {
  private static instance: StatusIconManager;
  private readonly statusBarItem: vscode.StatusBarItem;
  public state: "enabled" | "disabled" = "enabled";

  private constructor() {
    // Call the parent constructor
    super(() => this.statusBarItem.dispose());

    // Create the status bar item
    logger.info("StatusIconManager instance created");
    this.statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
    );

    // Initialize the status bar item
    this.initializeStatusBar();
    this.updateStatusBarIcon();

    // Update the status bar icon when the active text editor changes
    vscode.window.onDidChangeActiveTextEditor(() => this.updateStatusBarIcon());
  }

  /**
   * Get the singleton instance of StatusIconManager.
   * @returns {StatusIconManager} The singleton instance.
   */
  public static getInstance(): StatusIconManager {
    if (!StatusIconManager.instance) {
      // Create a new instance if not already created
      StatusIconManager.instance = new StatusIconManager();
      storage().context.subscriptions.push(StatusIconManager.instance);
      logger.debug("New StatusIconManager instance created");
    }
    return StatusIconManager.instance;
  }

  /**
   * Initialize the status bar item with default properties.
   */
  private initializeStatusBar(): void {
    logger.debug("Initializing status bar");
    this.statusBarItem.accessibilityInformation = {
      label: "Flexpilot Status",
    };
    this.statusBarItem.tooltip = "Flexpilot Status";
    this.statusBarItem.name = "Flexpilot Status";
    this.statusBarItem.command = "flexpilot.status.icon.menu";
    this.statusBarItem.show();
    logger.debug("Status bar initialized");
  }

  /**
   * Update the status bar icon based on the current state.
   */
  public updateStatusBarIcon(): void {
    if (this.isCompletionsEnabled()) {
      this.state = "enabled";
      this.statusBarItem.text = "$(flexpilot-default)";
    } else {
      this.state = "disabled";
      this.statusBarItem.text = "$(flexpilot-disabled)";
    }
  }

  /**
   * Check if inline completions are enabled or disabled.
   */
  public isCompletionsEnabled() {
    logger.debug("Updating status bar icon");
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      logger.debug("No active editor, setting status to disabled");
      return false;
    }
    const provider =
      ModelProviderManager.getInstance().getProvider("Inline Completion");
    if (!provider) {
      logger.debug("No provider found, setting status to disabled");
      return false;
    }
    const config = storage().get("completions.disabled.languages") || [];
    if (config.includes(editor.document.languageId)) {
      logger.debug("Language disabled, setting status to disabled");
      return false;
    }
    logger.debug("Setting status to enabled");
    return true;
  }

  /**
   * Reset the status bar item to default state.
   */
  public reset(): void {
    logger.debug("Resetting status bar");
    if (this.state === "disabled") {
      this.statusBarItem.text = "$(flexpilot-disabled)";
    } else {
      this.statusBarItem.text = "$(flexpilot-default)";
    }
  }

  /**
   * Set the status bar item to loading state.
   */
  public setLoading(): void {
    logger.debug("Setting status bar to loading");
    this.statusBarItem.text = "$(loading~spin)";
  }
}

// Export the StatusIconManager instance
export const statusIcon = () => StatusIconManager.getInstance();
