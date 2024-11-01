import * as vscode from "vscode";
import { LOCATIONS, PROVIDER_TYPES } from "../constants";
import { ILocationName, IModelType } from "../interfaces";
import { logger } from "../logger";
import { ModelProviderManager, ModelProviders } from "../providers";
import { storage } from "../storage";

/**
 * Represents a location pick-up item for a Quick Pick interface in VS Code.
 */
interface ILocationPickUpItem extends vscode.QuickPickItem {
  label: ILocationName;
  providerType: IModelType;
}

/**
 * Represents an item in a Quick Pick list with an optional provider ID.
 */
interface IModelPickUpItem extends vscode.QuickPickItem {
  providerId?: string;
}

/**
 * Represents an item in a Quick Pick list for selecting a model provider.
 */
interface IModelProviderPickUpItem extends vscode.QuickPickItem {
  ProviderClass: (typeof ModelProviders)[number] | undefined;
}

/**
 * ConfigureModelCommand class manages the model configuration functionality.
 * It implements the Singleton pattern to ensure a single instance across the application.
 */
export class ConfigureModelCommand extends vscode.Disposable {
  private static instance: ConfigureModelCommand;
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
      "flexpilot.configureModel",
      this.handler.bind(this),
    );
    logger.info("ConfigureModelCommand instance created");
  }

  /**
   * Gets the singleton instance of ConfigureModelCommand.
   * @returns {ConfigureModelCommand} The singleton instance.
   */
  public static register() {
    if (!ConfigureModelCommand.instance) {
      ConfigureModelCommand.instance = new ConfigureModelCommand();
      storage().context.subscriptions.push(ConfigureModelCommand.instance);
      logger.debug("New ConfigureModelCommand instance created");
    }
  }

  /**
   * Handles the model configuration process.
   * This method is the entry point for the configuration command.
   */
  public async handler(): Promise<void> {
    try {
      logger.info("Starting model configuration process");
      const quickPick = this.createQuickPick();

      // Handle button clicks
      quickPick.onDidTriggerItemButton(async (event) => {
        try {
          if (event.button.tooltip === "Configure") {
            await this.editModelConfiguration(event.item.label);
          } else if (event.button.tooltip === "Delete") {
            await this.deleteConfirmationDialog(event.item.label);
          }
        } catch (error) {
          logger.error(error as Error);
          logger.notifyError("Error in `onDidTriggerItemButton`");
        }
      });

      // Handle item selection
      quickPick.onDidChangeSelection(async (selection) => {
        try {
          if (selection[0].label === "Add new configuration") {
            await this.addNewConfiguration();
          } else if (selection[0].label === "Modify preferences") {
            await this.modifyPreferences();
          }
        } catch (error) {
          logger.error(error as Error);
          logger.notifyError("Error in `onDidTriggerItemButton`");
        }
      });

      // Dispose the quick pick on hide
      quickPick.onDidHide(() => {
        logger.debug("Quick pick was hidden");
        quickPick.dispose();
      });

      // Show the quick pick
      quickPick.show();
    } catch (error) {
      logger.error(error as Error);
      logger.notifyError("Error in `Configure Model` command");
    }
  }

  /**
   * Creates a QuickPick instance with the given items.
   * @returns {vscode.QuickPick<vscode.QuickPickItem>} The created QuickPick instance.
   */
  private createQuickPick(): vscode.QuickPick<vscode.QuickPickItem> {
    logger.debug("Creating quick pick for model configuration");
    const quickPickItems: vscode.QuickPickItem[] = [
      {
        label: "Configurations",
        kind: vscode.QuickPickItemKind.Separator,
      },
    ];

    // Add existing configurations
    let hasConfigurations = false;
    for (const nickname of storage().models.list()) {
      const config = storage().models.get(nickname);
      if (!config) {
        logger.warn(`No configuration found for nickname: ${nickname}`);
        continue;
      }
      const ModelProvider = ModelProviders.find(
        (item) => item.providerId === config.providerId,
      );
      if (!ModelProvider) {
        logger.warn(
          `No ModelProvider found for providerId: ${config.providerId}`,
        );
        continue;
      }
      hasConfigurations = true;
      quickPickItems.push({
        detail: `${config.model}`,
        buttons: [
          {
            iconPath: new vscode.ThemeIcon("gear"),
            tooltip: "Configure",
          },
          {
            iconPath: new vscode.ThemeIcon("trash"),
            tooltip: "Delete",
          },
        ],
        label: nickname,
        alwaysShow: true,
        description: `(${ModelProvider.providerName})`,
      });
    }

    // Add menu options
    quickPickItems.push(
      { label: "Menu Options", kind: vscode.QuickPickItemKind.Separator },
      {
        label: "Add new configuration",
        detail: "Add a new language model provider to the configuration",
      },
      {
        label: "Modify preferences",
        detail: "Set your language model preferences based on location",
      },
    );

    // Create the quick pick
    const quickPick = vscode.window.createQuickPick();
    quickPick.items = quickPickItems;
    quickPick.title = "Flexpilot: Configure the Language Model Provider";
    quickPick.ignoreFocusOut = true;
    quickPick.canSelectMany = false;
    quickPick.placeholder = hasConfigurations
      ? "Select a language model provider to edit"
      : "Please add a new language model provider";

    return quickPick;
  }

  /**
   * Modifies the usage preferences for different locations.
   * This method allows users to set which model should be used for each location.
   */
  private async modifyPreferences(): Promise<void> {
    logger.info("Starting preference modification process");

    // Get the location to modify
    const locationQuickPickItems: ILocationPickUpItem[] = LOCATIONS.map(
      (location) => {
        const preference = storage().usage.get(location.name);
        return {
          providerType: location.type,
          label: location.name,
          detail: preference
            ? `Currently uses \`${preference.nickname}\``
            : "Currently Disabled",
          description: `(${location.description})`,
        };
      },
    );
    const pickedLocation = await vscode.window.showQuickPick(
      locationQuickPickItems,
      {
        title: "Flexpilot: Modify Model Usage Preferences",
        ignoreFocusOut: true,
        canPickMany: false,
        placeHolder: "Select the usage location",
      },
    );
    if (!pickedLocation) {
      logger.info("Location selection cancelled");
      return;
    }

    // Get the model to be used in the location
    const preference = storage().usage.get(pickedLocation.label);
    const modelQuickPickItems: IModelPickUpItem[] = [
      {
        label: "Other commands",
        kind: vscode.QuickPickItemKind.Separator,
      },
      { label: `Disable \`${pickedLocation.label}\`` },
    ];
    for (const nickname of storage().models.list()) {
      const config = storage().models.get(nickname);
      if (!config) {
        continue;
      }
      const ModelProvider = ModelProviders.find(
        (item) => item.providerId === config.providerId,
      );
      if (
        !ModelProvider ||
        ModelProvider.providerType !== pickedLocation.providerType
      ) {
        continue;
      }
      modelQuickPickItems.unshift({
        providerId: config.providerId,
        label: nickname,
        description:
          preference?.nickname === nickname
            ? "(Currently Selected)"
            : undefined,
        detail: config.model,
      });
    }
    const pickedModel = await vscode.window.showQuickPick(modelQuickPickItems, {
      title: "Flexpilot: Modify Model Usage Preferences",
      ignoreFocusOut: true,
      canPickMany: false,
      placeHolder: `Select the model config to be used in \`${pickedLocation.label}\``,
    });
    if (!pickedModel) {
      logger.info("Model selection cancelled");
      return;
    }

    // Update the usage preference
    if (pickedModel.label === `Disable \`${pickedLocation.label}\``) {
      await storage().usage.set(pickedLocation.label, undefined);
      logger.notifyInfo(
        `Model usage for \`${pickedLocation.label}\` is disabled`,
      );
    } else if (pickedModel.providerId) {
      await storage().usage.set(pickedLocation.label, {
        providerId: pickedModel.providerId,
        nickname: pickedModel.label,
        locationName: pickedLocation.label,
      });
      logger.notifyInfo(
        `Model \`${pickedModel.label}\` is successfully set to be used in \`${pickedLocation.label}\``,
      );
    }
    await ModelProviderManager.getInstance().initProviders();
  }

  /**
   * Adds a new model configuration.
   * This method guides the user through the process of adding a new model provider.
   */
  private async addNewConfiguration(): Promise<void> {
    logger.info("Starting process to add new model configuration");

    // Select a model provider
    const providerOptions = PROVIDER_TYPES.map((providerType) => {
      const output: IModelProviderPickUpItem[] = [];
      output.push({
        description: `${providerType} Models`,
        label: `${providerType} Models`,
        ProviderClass: undefined,
        kind: vscode.QuickPickItemKind.Separator,
      });
      for (const ModelProvider of ModelProviders) {
        if (ModelProvider.providerType === providerType) {
          output.push({
            label: ModelProvider.providerName,
            ProviderClass: ModelProvider,
            description: `(${ModelProvider.providerId})`,
          });
        }
      }
      return output;
    }).flat();
    const selected = await vscode.window.showQuickPick(providerOptions, {
      ignoreFocusOut: true,
      canPickMany: false,
      title: "Select Language Model Provider",
      placeHolder: "Choose the language model provider",
    });
    if (!selected?.ProviderClass) {
      logger.info("Model provider selection cancelled");
      return;
    }

    // Get a nickname for the new configuration
    const nickname = await vscode.window.showInputBox({
      placeHolder: "Enter a nickname for this provider (e.g., Azure GPT-4)",
      ignoreFocusOut: true,
      title: "Add New Provider",
      validateInput: (value) => {
        const isDuplicate = storage()
          .models.list()
          .map((nickname) => nickname.trim().toLowerCase())
          .includes(value.trim().toLowerCase());
        return isDuplicate ? "Nickname already exists" : undefined;
      },
    });
    if (!nickname) {
      logger.info("Nickname input cancelled");
      return;
    }

    // Configure the new provider
    try {
      await selected.ProviderClass.configure(nickname);
    } catch (error) {
      // Log the error and notify the user
      logger.error(error as Error);
      logger.notifyError("Error in configuring the model provider");
      return;
    }

    // Notify the user about the successful configuration
    logger.notifyInfo(
      `Configured \`${selected.ProviderClass.providerName}\` model provider successfully`,
    );

    // Ask for the locations to use the new model
    const selectedLocationType = selected.ProviderClass.providerType;

    const locationQuickPickItems: ILocationPickUpItem[] = LOCATIONS.filter(
      (location) => location.type === selectedLocationType,
    ).map((location) => {
      const preference = storage().usage.get(location.name);
      return {
        providerType: location.type,
        label: location.name,
        detail: preference
          ? `Currently uses \`${preference.nickname}\``
          : "Currently Disabled",
        description: `(${location.description})`,
      };
    });

    const pickedLocations = await vscode.window.showQuickPick(
      locationQuickPickItems,
      {
        title: "Flexpilot: Configure Model Usage",
        ignoreFocusOut: true,
        canPickMany: true,
        placeHolder: "Select the locations where this model will be used",
      },
    );

    if (pickedLocations === undefined) {
      throw new Error("Location selection cancelled");
    }

    // Update the usage preferences
    for (const pickedLocation of pickedLocations) {
      storage().usage.set(pickedLocation.label, {
        providerId: selected.ProviderClass.providerId,
        nickname: nickname,
        locationName: pickedLocation.label,
      });
      await ModelProviderManager.getInstance().initProviders();
    }

    // Set the context to indicate successful configuration for walkthroughs
    await vscode.commands.executeCommand(
      "setContext",
      "flexpilot:walkthroughConfigureModel",
      true,
    );

    logger.notifyInfo(
      `Model \`${nickname}\` is successfully set to be used in the selected locations`,
    );
  }

  /**
   * Edits an existing model configuration.
   * @param {string} nickname - The nickname of the configuration to edit.
   */
  private async editModelConfiguration(nickname: string): Promise<void> {
    logger.info(`Starting to edit existing model configuration: ${nickname}`);
    const config = storage().models.get(nickname);
    if (!config) {
      logger.warn(`No configuration found for nickname: ${nickname}`);
      return;
    }
    const ModelProvider = ModelProviders.find(
      (item) => item.providerId === config.providerId,
    );
    if (!ModelProvider) {
      logger.warn(
        `No ModelProvider found for providerId: ${config.providerId}`,
      );
      return;
    }

    // Configure the model provider
    try {
      await ModelProvider.configure(nickname);
    } catch (error) {
      // Log the error and notify the user
      logger.error(error as Error);
      logger.notifyError("Error in configuring the model provider");
      return;
    }

    await ModelProviderManager.getInstance().initProviders();
    logger.notifyInfo(`Updated \`${nickname}\` configuration successfully`);
  }

  /**
   * Shows a confirmation dialog for deleting a configuration.
   * @param {string} nickname - The nickname of the configuration to delete.
   */
  private async deleteConfirmationDialog(nickname: string): Promise<void> {
    logger.info(
      `Initiating delete confirmation for configuration: ${nickname}`,
    );
    const confirmNickname = await vscode.window.showInputBox({
      title: "Flexpilot: Delete language model provider",
      ignoreFocusOut: true,
      prompt: "Enter nickname here to confirm",
      validateInput(value) {
        return value === nickname
          ? undefined
          : `Text does not match with \`${nickname}\``;
      },
      placeHolder: `Enter \`${nickname}\` to confirm deletion`,
    });
    // Delete the configuration if the input matches
    if (confirmNickname === nickname) {
      await storage().models.set(nickname);
      for (const location of LOCATIONS) {
        // Remove the preference if it matches the deleted nickname
        const preference = storage().usage.get(location.name);
        if (preference?.nickname === nickname) {
          await storage().usage.set(location.name, undefined);
        }
      }
      // Reinitialize the model providers
      await ModelProviderManager.getInstance().initProviders();
      logger.notifyInfo(`Deleted model configuration: \`${nickname}\``);
    } else {
      logger.info(
        `Deletion of configuration ${nickname} was cancelled or input didn't match`,
      );
    }
  }
}
