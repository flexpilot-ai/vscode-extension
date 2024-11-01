import { CoreMessage } from "ai";
import React from "react";
import ReactDOMServer from "react-dom/server";
import TurndownService from "turndown";
import * as vscode from "vscode";

/**
 * Props for the Message component.
 */
interface MessageProps {
  role: "assistant" | "user" | "system";
  children: React.ReactNode;
  reference?: vscode.Uri | vscode.Location; // NOSONAR
}

/**
 * Props for the Code component.
 */
interface CodeProps {
  children: React.ReactNode;
  language?: string;
}

/**
 * Message component for rendering chat messages.
 */
export const Message: React.FC<MessageProps> = ({ children, role }) => (
  <div data-role={role}>{children}</div>
);

/**
 * Code component for rendering code blocks.
 */
export const Code: React.FC<CodeProps> = ({ children, language }) => (
  <pre>
    <code className={language ? `language-${language}` : ""}>{children}</code>
  </pre>
);

// Configure TurndownService for consistent Markdown conversion
const turndownService = new TurndownService({
  headingStyle: "atx",
  hr: "---",
  bulletListMarker: "-",
  codeBlockStyle: "fenced",
  fence: "```",
  emDelimiter: "*",
  strongDelimiter: "**",
  linkStyle: "inlined",
  linkReferenceStyle: "full",
  preformattedCode: true,
});

/**
 * Converts a JSX Message element to a core message object.
 * @param jsx - The JSX Message element to convert.
 * @returns A core message object.
 * @throws {Error} if the JSX element is invalid or has an invalid role.
 */
export const jsxToCoreMessage = (jsx: React.ReactElement): CoreMessage => {
  const html = ReactDOMServer.renderToStaticMarkup(jsx);
  if (jsx.type !== Message) {
    throw new Error("Invalid JSX element: expected Message component");
  }
  const content = turndownService.turndown(html);
  switch (jsx.props.role) {
    case "assistant":
    case "user":
    case "system":
      return { role: jsx.props.role, content: content };
    default:
      throw new Error(`Invalid role in JSX element: ${jsx.props.role}`);
  }
};

/**
 * Converts a JSX element to a VS Code MarkdownString.
 * @param jsx - The JSX element to convert.
 * @returns A VS Code MarkdownString representation of the JSX element.
 */
export const jsxToMarkdown = (
  jsx: React.ReactElement,
): vscode.MarkdownString => {
  return new vscode.MarkdownString(
    turndownService.turndown(ReactDOMServer.renderToStaticMarkup(jsx)),
  );
};
