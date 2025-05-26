import Markdown from "react-markdown";
import { Link } from "react-router";
import smartypants from "remark-smartypants";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeKatex from "rehype-katex";
import rehypeStringify from "rehype-stringify";

export default function MyMarkdown({ children }: { children: string }) {
  return (
    <div className="artintellica-prose">
      <Markdown
        remarkPlugins={[
          smartypants,
          remarkParse,
          remarkMath,
          remarkGfm,
          remarkRehype,
        ]}
        rehypePlugins={[rehypeKatex, rehypeStringify]}
        components={{
          a: ({ children, href }) => {
            return (
              <Link
                to={href || "#"}
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
