import Markdown from "react-markdown";
import { Link } from "react-router";
import smartypants from "remark-smartypants";
import remarkGfm from "remark-gfm";

export default function MyMarkdown({ children }: { children: string }) {
  return (
    <div className="artintellica-prose">
      <Markdown
        remarkPlugins={[smartypants]}
        components={{
          a: ({ children, href }) => {
            return (
              <Link
                to={
                  href
                    ? href.startsWith("./")
                      ? `.${href}` // force remix to handle relative paths correctly
                      : href
                    : href || ""
                }
                onClick={(e: React.MouseEvent<HTMLAnchorElement>) => {
                  e.stopPropagation();
                }}
              >
                {children}
              </Link>
            );
          },
        }}
      >
        {children}
      </Markdown>
    </div>
  );
}
