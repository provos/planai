import { visit } from 'unist-util-visit';

/**
 * Remark plugin to inject raw markdown content into frontmatter
 */
export function injectRawContent() {
  return function transformer(tree, file) {
    // Get the raw markdown content
    const rawContent = file.value;
    
    // Find the frontmatter node
    visit(tree, 'yaml', (node) => {
      // Inject the raw content into the frontmatter
      // This will be available in the Astro component
      file.data.astro.frontmatter._rawContent = rawContent;
    });
    
    return tree;
  };
}