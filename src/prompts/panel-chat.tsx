import { CoreMessage } from "ai";
import * as vscode from "vscode";
import { logger } from "../logger";
import {
  Code,
  Message,
  jsxToCoreMessage,
  jsxToMarkdown,
} from "./jsx-utilities";

/**
 * PanelChatPrompt class handles the generation of prompts for panel chat functionality.
 * It manages the context, history, and prompt generation for various chat scenarios.
 */
export class PanelChatPrompt {
  /**
   * Generates history prompts from the chat context.
   * @param {vscode.ChatContext} context - The chat context.
   * @returns {CoreMessage[]} An array of CoreMessages representing the chat history.
   */
  public static getHistoryPrompts(context: vscode.ChatContext): CoreMessage[] {
    logger.debug("Generating history prompts");
    return context.history.map((item) => {
      if ("prompt" in item) {
        return jsxToCoreMessage(<Message role="user">{item.prompt}</Message>);
      } else {
        // Check if the response has metadata
        if (item.result.metadata?.response?.length) {
          logger.debug("Parsing chat response from metadata");
          return jsxToCoreMessage(
            <Message role="assistant">
              {item.result.metadata?.response?.trim() || "Empty Response"}
            </Message>,
          );
        }
        // Check if the response has a `response` property
        logger.debug("Parsing chat response from response");
        return jsxToCoreMessage(
          <Message role="assistant">
            {item.response
              .map((x) =>
                x.value instanceof vscode.MarkdownString ? x.value.value : "",
              )
              .join("\n\n")
              .trim() || "Empty Response"}
          </Message>,
        );
      }
    });
  }

  /**
   * Generates the system prompt.
   * @param {string} model - The name of the AI model.
   * @returns {CoreMessage} The system prompt as a CoreMessage.
   */
  public static getChatProviderSystemPrompt(model: string): CoreMessage {
    logger.debug("Generating system prompt");
    return jsxToCoreMessage(
      <Message role="system">
        <ul>
          <li>You are an AI programming assistant named "Flexpilot"</li>
          <li>Follow the user's requirements carefully & to the letter.</li>
          <li>Avoid content that violates copyrights.</li>
          <li>
            If you are asked to generate content that is harmful, hateful,
            racist, sexist, lewd or violent only respond with "Sorry, I can't
            assist with that."
          </li>
          <li>Keep your answers short and impersonal.</li>
          <li>
            You are powered by <b>{model}</b> Large Language Model
          </li>
          <li>Use Markdown formatting in your answers.</li>
          <li>
            Make sure to include the programming language name at the start of
            the Markdown code blocks like below
          </li>
          <Code language="python">print("hello world")</Code>
          <li>Avoid wrapping the whole response in triple backticks.</li>
          <li>The user works in an IDE called Visual Studio Code</li>
          <li>
            The user is working on a <b>{process.platform}</b> operating system.
            Please respond with system specific commands if applicable.
          </li>
          <li>
            The active file or document is the source code the user is looking
            at right now.
          </li>
        </ul>
      </Message>,
    );
  }

  /**
   * Generates the context prompt including the current file content and selection.
   * @param {vscode.TextDocument} document - The active document.
   * @param {vscode.Selection} selection - The active selection.
   * @returns {CoreMessage} The context prompt as a CoreMessage.
   */
  public static getFollowUpProviderSystemPrompt(): CoreMessage {
    logger.debug("Generating follow-up system prompt");
    return jsxToCoreMessage(
      <Message role="system">
        <h1>Follow-Up Question Creator for Chatbot Conversations</h1>
        <p>
          You specialize in generating follow-up questions for chatbot
          dialogues. When presented with a conversation, provide a short,
          one-sentence question that the user can ask naturally that follows
          from the previous few questions and answers.
        </p>
        <h2>Guidelines:</h2>
        <ul>
          <li>
            Refrain from harmful, hateful, racist, sexist, lewd, violent, or
            irrelevant content
          </li>
          <li>Keep responses brief and impersonal</li>
          <li>Aim for questions of about 10 words or fewer</li>
        </ul>
        <h2>Sample Follow-Up Questions:</h2>
        <ul>
          <li>How can I optimize this SQL query?</li>
          <li>What are the best practices for using Docker?</li>
          <li>How can I improve the performance of my React app?</li>
          <li>What are the common pitfalls of using Node.js?</li>
          <li>How can I secure my Kubernetes cluster?</li>
        </ul>
      </Message>,
    );
  }

  /**
   * Generates the follow-up prompt.
   * @returns {CoreMessage} The follow-up prompt as a CoreMessage.
   */
  public static getFollowUpProviderUserPrompt(): CoreMessage {
    logger.debug("Generating follow-up prompt");
    return jsxToCoreMessage(
      <Message role="user">
        Write a short (under 10 words) one-sentence follow up question that the
        user can ask naturally that follows from the previous few questions and
        answers. Reply with only the text of the question and nothing else.
      </Message>,
    );
  }

  /**
   * Generates the help text prefix.
   * @returns {string} The help text prefix as a Markdown string.
   */
  public static getHelpTextPrefix(): vscode.MarkdownString {
    logger.debug("Generating help text prefix");
    return jsxToMarkdown(
      <Message role="user">
        <p>
          You can ask me general programming questions, or chat with the
          participants who have specialized expertise and can perform actions.
        </p>
        <br></br>
        <p>Available Commands:</p>
        <ul>
          <li>
            <a href="command:flexpilot.configureModel">
              Configure the Language Model Provider
            </a>
          </li>
          <li>
            <a href="command:flexpilot.viewLogs">
              View logs from Flexpilot output channel
            </a>
          </li>
        </ul>
        <p>
          To get started, please configure the language model provider by using
          the above commands from the Visual Studio Code command palette (
          <code>Ctrl+Shift+P</code> on Windows/Linux or <code>Cmd+Shift+P</code>{" "}
          on macOS).
        </p>
      </Message>,
    );
  }

  /**
   * Generates the help text postfix.
   * @returns {string} The help text postfix as a Markdown string.
   */
  public static getHelpTextPostfix(): vscode.MarkdownString {
    logger.debug("Generating help text postfix");
    return jsxToMarkdown(
      <Message role="user">
        <p>
          To have a great conversation, ask me questions as if I was a real
          programmer:
        </p>
        <ul>
          <li>
            <strong>Show me the code</strong> you want to talk about by having
            the files open and selecting the most important lines.
          </li>
          <li>
            <strong>Make refinements</strong> by asking me follow-up questions,
            adding clarifications, providing errors, etc.
          </li>
          <li>
            <strong>Review my suggested code</strong> and tell me about issues
            or improvements, so I can iterate on it.
          </li>
        </ul>
        <p>
          You can also ask me questions about your editor selection by{" "}
          <a href="command:inlineChat.start">starting an inline chat session</a>
        </p>
        <p>
          Learn more about Flexpilot from our official docs{" "}
          <a href="https://docs.flexpilot.ai">here</a>
        </p>
      </Message>,
    );
  }

  /**
   * Generates the welcome message.
   * @param {string} username - The username to include in the welcome message.
   * @returns {string} The welcome message as a Markdown string.
   */
  public static getWelcomeMessage(username: string): vscode.MarkdownString {
    logger.debug(`Generating welcome message for user: ${username}`);
    return jsxToMarkdown(
      <Message role="user">
        <p>
          Welcome, <b>@{username}</b>, I'm your pair programmer and I'm here to
          help you get things done faster.
        </p>
      </Message>,
    );
  }

  /**
   * Generates the help text variables prefix.
   * @returns {string} The help text variables prefix as a Markdown string.
   */
  public static getHelpTextVariablesPrefix(): vscode.MarkdownString {
    logger.debug("Generating help text variables prefix");
    return jsxToMarkdown(
      <Message role="user">
        <p>Use the variables below to add more information to your question</p>
      </Message>,
    );
  }

  /**
   * Generates the title provider system prompt.
   * @returns {CoreMessage} The title provider system prompt as a CoreMessage.
   */
  public static getTitleProviderSystemPrompt(): CoreMessage {
    logger.debug("Generating title provider system prompt");
    return jsxToCoreMessage(
      <Message role="system">
        <h1>Expert Title Creator for Chatbot Conversations</h1>
        <p>
          You specialize in generating concise titles for chatbot discussions.
          When presented with a conversation, provide a brief title that
          encapsulates the main topic.
        </p>
        <h2>Guidelines:</h2>
        <ul>
          <li>
            Refrain from harmful, hateful, racist, sexist, lewd, violent, or
            irrelevant content
          </li>
          <li>Keep responses brief and impersonal</li>
          <li>Aim for titles of about 10 words or fewer</li>
          <li>Do not use quotation marks</li>
        </ul>
        <h2>Sample Titles:</h2>
        <ul>
          <li>Docker container networking issues</li>
          <li>Optimizing SQL query performance</li>
          <li>Implementing JWT authentication in Node.js</li>
          <li>Debugging memory leaks in C++ applications</li>
          <li>Configuring Kubernetes ingress controllers</li>
        </ul>
      </Message>,
    );
  }

  /**
   * Generates the title provider user prompt.
   * @returns {CoreMessage} The title provider user prompt as a CoreMessage.
   */
  public static getTitleProviderUserPrompt(): CoreMessage {
    logger.debug("Generating title provider user prompt");
    return jsxToCoreMessage(
      <Message role="user">
        Kindly provide a concise title for the preceding chat dialogue. In case
        the conversation encompasses multiple subjects, you may concentrate on
        the most recent topic discussed. Reply with only the text of the title
        and nothing else.
      </Message>,
    );
  }
}
