import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Astro integration to generate llms.txt files from markdown content
 */
export function generateLlmsTxt() {
  return {
    name: 'generate-llms-txt',
    hooks: {
      'astro:build:done': async ({ dir, pages }) => {
        console.log('Generating llms.txt files...');
        
        const docsDir = path.join(__dirname, '../../src/content/docs');
        const outputDir = path.join(dir.pathname, 'llms');
        
        // Ensure output directory exists
        await fs.mkdir(outputDir, { recursive: true });
        
        // Process all markdown files
        await processDirectory(docsDir, docsDir, outputDir);
        
        console.log('llms.txt files generated successfully!');
      }
    }
  };
}

async function processDirectory(currentDir, baseDir, outputDir) {
  const entries = await fs.readdir(currentDir, { withFileTypes: true });
  
  for (const entry of entries) {
    const fullPath = path.join(currentDir, entry.name);
    
    if (entry.isDirectory()) {
      // Recursively process subdirectories
      await processDirectory(fullPath, baseDir, outputDir);
    } else if (entry.name.endsWith('.md') || entry.name.endsWith('.mdx')) {
      // Process markdown files
      await processMarkdownFile(fullPath, baseDir, outputDir);
    }
  }
}

async function processMarkdownFile(filePath, baseDir, outputDir) {
  try {
    // Read the markdown content
    const content = await fs.readFile(filePath, 'utf-8');
    
    // Calculate the relative path and output filename
    const relativePath = path.relative(baseDir, filePath);
    const slug = relativePath
      .replace(/\.(md|mdx)$/, '')
      .replace(/\\/g, '/')
      .replace(/\/index$/, '');
    
    // Create the output path
    const outputPath = path.join(outputDir, `${slug}.txt`);
    
    // Ensure the directory exists
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    
    // Extract frontmatter
    const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
    
    if (frontmatterMatch) {
      const [, frontmatter, body] = frontmatterMatch;
      
      // Parse frontmatter to get title and description
      const titleMatch = frontmatter.match(/title:\s*(.+)/);
      const descMatch = frontmatter.match(/description:\s*(.+)/);
      
      const title = titleMatch ? titleMatch[1].replace(/^["']|["']$/g, '') : '';
      const description = descMatch ? descMatch[1].replace(/^["']|["']$/g, '') : '';
      
      // Create the llms.txt content with metadata
      let llmsContent = `# ${title}\n\n`;
      if (description) {
        llmsContent += `${description}\n\n`;
      }
      llmsContent += `---\n\n`;
      llmsContent += body.trim();
      
      // Write the file
      await fs.writeFile(outputPath, llmsContent, 'utf-8');
    } else {
      // No frontmatter, just copy the content
      await fs.writeFile(outputPath, content, 'utf-8');
    }
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error);
  }
}