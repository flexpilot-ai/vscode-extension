import * as vscode from "vscode";
import { logger } from "./logger";
import { storage } from "./storage";

/**
 * EventsManager class to handle VSCode extension events and actions.
 * This class follows the Singleton pattern to ensure only one instance exists throughout the application lifecycle.
 */
class EventsManager extends vscode.Disposable {
  private static instance: EventsManager;
  private readonly disposables: vscode.Disposable[] = [];
  private lastTerminalExecutedCommand:
    | vscode.TerminalExecutedCommand
    | undefined;

  /**
   * Creates a new instance of EventsManager.
   */
  private constructor() {
    // Call the parent constructor
    super(() => this.disposables.forEach((disposable) => disposable.dispose()));
    // Initialize all event listeners
    this.initializeEventListeners();
  }

  /**
   * Gets the last terminal execution details.
   */
  public getLastTerminalExecutedCommand() {
    return this.lastTerminalExecutedCommand;
  }

  /**
   * Gets the singleton instance of EventsManager.
   * @returns {EventsManager} The singleton instance.
   */
  public static getInstance(): EventsManager {
    if (!EventsManager.instance) {
      // Create a new instance if not already created
      EventsManager.instance = new EventsManager();
      storage().context.subscriptions.push(EventsManager.instance);
      logger.debug("New EventsManager instance created");
    }
    return EventsManager.instance;
  }

  /**
   * Initializes all event listeners.
   */
  private initializeEventListeners(): void {
    this.disposables.push(
      vscode.window.onDidExecuteTerminalCommand(
        (event: vscode.TerminalExecutedCommand) => {
          logger.debug(`Terminal command executed: ${event.commandLine}`);
          this.lastTerminalExecutedCommand = event;
        },
      ),
    );
    logger.info("Event listeners registered successfully");
  }

  /**
   * Disposes of all registered event listeners.
   */
  public dispose(): void {
    this.disposables.forEach((disposable) => disposable.dispose());
    logger.info("All event listeners disposed successfully");
  }
}

// Export the EventsManager instance
export const eventsManager = () => EventsManager.getInstance();
