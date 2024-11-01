import * as vscode from "vscode";
import { logger } from "../logger";
import { statusIcon } from "../status-icon";
import { storage } from "../storage";

/**
 * Extends vscode.QuickPickItem with additional properties for custom functionality.
 */
interface ICustomQuickPickItem extends vscode.QuickPickItem {
  status?: string;
  handler?: () => Promise<void>;
}

/**
 * StatusIconMenuCommand class manages the status icon menu functionality.
 * It implements the Singleton pattern to ensure a single instance across the application.
 */
export class StatusIconMenuCommand extends vscode.Disposable {
  private static instance: StatusIconMenuCommand;
  private readonly disposable: vscode.Disposable;

  /**
   * Private constructor to prevent direct instantiation.
   * Registers the command and initializes the disposable.
   */
  private constructor() {
    // Call the parent constructor
    super(() => this.disposable.dispose());

    // Register the command
    this.disposable = vscode.commands.registerCommand(
      "flexpilot.status.icon.menu",
      this.handler.bind(this),
    );
    logger.info("StatusIconMenuCommand instance created");
  }

  /**
   * Gets the singleton instance of StatusIconMenuCommand.
   * @returns {StatusIconMenuCommand} The singleton instance.
   */
  public static register() {
    if (!StatusIconMenuCommand.instance) {
      StatusIconMenuCommand.instance = new StatusIconMenuCommand();
      storage().context.subscriptions.push(StatusIconMenuCommand.instance);
      logger.debug("New StatusIconMenuCommand instance created");
    }
  }

  /**
   * Handles the status bar icon menu functionality.
   * Displays a quick pick menu with various options and executes the selected action.
   */
  public async handler(): Promise<void> {
    try {
      logger.info("Handling `StatusIconMenuCommand`");
      const menuItems: ICustomQuickPickItem[] = this.createMenuItems();
      const selectedMenu = await vscode.window.showQuickPick(menuItems, {
        placeHolder: "Select an option",
        ignoreFocusOut: true,
        title: "Flexpilot Completions Menu",
      });

      if (selectedMenu?.handler) {
        await selectedMenu.handler();
      }
    } catch (error) {
      logger.error(error as Error);
      logger.notifyError("Error in `Status Icon Menu` command");
    }
  }

  /**
   * Creates the menu items for the quick pick menu.
   * @returns {ICustomQuickPickItem[]} An array of custom quick pick items.
   */
  private createMenuItems(): ICustomQuickPickItem[] {
    let menuItems: ICustomQuickPickItem[] = [];

    // Add status item
    menuItems.push(this.createStatusMenuItem());

    // Add separator
    menuItems.push({ label: "", kind: vscode.QuickPickItemKind.Separator });

    // Add language-specific item if there's an active text editor
    const activeTextEditor = vscode.window.activeTextEditor;
    if (activeTextEditor) {
      menuItems.push(this.getLanguageSpecificMenuItem(activeTextEditor));
    }

    // Add separator and general menu items
    menuItems.push({ label: "", kind: vscode.QuickPickItemKind.Separator });

    // Add general menu items
    menuItems = menuItems.concat(this.createGeneralMenuItems());

    return menuItems;
  }

  /**
   * Creates the status menu item based on the current configuration and state.
   * @returns {ICustomQuickPickItem} The status menu item.
   */
  private createStatusMenuItem(): ICustomQuickPickItem {
    if (!storage().usage.get("Inline Completion")) {
      return {
        label: "$(flexpilot-default) Status: Inline Completion Not Configured",
      };
    }
    return statusIcon().state === "disabled"
      ? { label: "$(flexpilot-default) Status: Disabled" }
      : { label: "$(flexpilot-default) Status: Ready" };
  }

  /**
   * Creates general menu items for common actions.
   * @returns {ICustomQuickPickItem[]} An array of general menu items.
   */
  private createGeneralMenuItems(): ICustomQuickPickItem[] {
    return [
      {
        label: "$(keyboard) Edit Keyboard Shortcuts...",
        handler: async () => {
          vscode.commands.executeCommand(
            "workbench.action.openGlobalKeybindings",
          );
        },
      },
      {
        label: "$(settings-gear) Edit Settings...",
        handler: async () => {
          vscode.commands.executeCommand(
            "workbench.action.openSettings",
            "flexpilot",
          );
        },
      },
      {
        label: "$(chat-editor-label-icon) Open Flexpilot Chat",
        handler: async () => {
          vscode.commands.executeCommand(
            "workbench.panel.chat.view.copilot.focus",
          );
        },
      },
      {
        label: "$(remote-explorer-documentation) View Flexpilot Docs...",
        handler: async () => {
          vscode.env.openExternal(
            vscode.Uri.parse("https://docs.flexpilot.ai"),
          );
        },
      },
    ];
  }

  /**
   * Creates a language-specific menu item for enabling or disabling completions.
   * @param {vscode.TextEditor} activeTextEditor The currently active text editor.
   * @returns {ICustomQuickPickItem} A menu item for toggling completions for the current language.
   */
  private getLanguageSpecificMenuItem(
    activeTextEditor: vscode.TextEditor,
  ): ICustomQuickPickItem {
    const config = storage().get("completions.disabled.languages") || [];
    const languageId = activeTextEditor.document.languageId;
    const isDisabled = config.includes(languageId);

    return {
      label: `\`${
        isDisabled ? "Enable" : "Disable"
      }\` Completions for \`${languageId}\``,
      handler: async () => {
        if (isDisabled) {
          config.splice(config.indexOf(languageId), 1);
        } else {
          config.push(languageId);
        }
        await storage().set("completions.disabled.languages", config);
        statusIcon().updateStatusBarIcon();
      },
    };
  }
}
