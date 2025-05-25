import {
  Link,
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  isRouteErrorResponse,
  useRouteError,
} from "react-router";
import "./tailwind.css";
import type { Route } from "./+types/root.js";

export const meta: Route.MetaFunction = () => {
  return [
    { title: "Artintellica" },
    {
      name: "description",
      content: "Open-source educational resources for AI.",
    },
  ];
};

export default function App() {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body className="bg-gray-100 text-black dark:bg-gray-900 dark:text-white">
        <Outlet />
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}
