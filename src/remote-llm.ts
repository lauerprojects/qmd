import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
  QueryType
} from "./llm-types.js";

import { getModel, complete, type Model, type Context, type AssistantMessage } from "@mariozechner/pi-ai";

export type RemoteLLMConfig = {
  apiKey: string;
  baseURL?: string;
  embedModel?: string;
  generateModel?: string;
  rerankModel?: string;
  timeoutMs?: number;
};

const DEFAULT_BASE_URL = "https://openrouter.ai/api/v1";
const DEFAULT_TIMEOUT_MS = 60000;

export class RemoteLLM implements LLM {
  private apiKey: string;
  private baseURL: string;
  private embedModel: string;
  private generateModel: string;
  private rerankModel?: string;
  private timeoutMs: number;

  constructor(config: RemoteLLMConfig) {
    this.apiKey = config.apiKey;
    this.baseURL = config.baseURL || DEFAULT_BASE_URL;
    // Remove trailing slash if present
    if (this.baseURL.endsWith('/')) {
      this.baseURL = this.baseURL.slice(0, -1);
    }
    this.embedModel = config.embedModel || "text-embedding-3-small";
    this.generateModel = config.generateModel || "openai/gpt-3.5-turbo";
    this.rerankModel = config.rerankModel;
    this.timeoutMs = config.timeoutMs || DEFAULT_TIMEOUT_MS;
  }

  private async fetchAPI(endpoint: string, body: any): Promise<any> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${this.apiKey}`,
          // OpenRouter specific headers
          "HTTP-Referer": "https://github.com/tobi/qmd",
          "X-Title": "QMD"
        },
        body: JSON.stringify(body),
        signal: controller.signal
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API request failed: ${response.status} ${response.statusText} - ${errorText}`);
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      const model = options.model || this.embedModel;
      const response = await this.fetchAPI("/embeddings", {
        model,
        input: text
      });

      if (!response.data || response.data.length === 0) {
        return null;
      }

      return {
        embedding: response.data[0].embedding,
        model
      };
    } catch (error) {
      console.error("Remote embedding error:", error);
      return null;
    }
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    try {
      // OpenAI API supports batch embeddings
      const response = await this.fetchAPI("/embeddings", {
        model: this.embedModel,
        input: texts
      });

      if (!response.data) {
        return texts.map(() => null);
      }

      // Sort by index to ensure order matches input
      const sortedData = response.data.sort((a: any, b: any) => a.index - b.index);
      
      return sortedData.map((item: any) => ({
        embedding: item.embedding,
        model: this.embedModel
      }));
    } catch (error) {
      console.error("Remote batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  /**
   * Resolve the pi-ai model from the configured string.
   * Handles "provider/modelId" or defaults to "openai/modelId".
   */
  private resolvePiModel(modelStr: string): Model<any> {
    let provider = 'openai';
    let modelId = modelStr;

    if (modelStr.includes('/')) {
      const parts = modelStr.split('/');
      provider = parts[0]!;
      modelId = parts.slice(1).join('/');
    }

    // Special handling for custom base URL (e.g. OpenRouter via OpenAI compat)
    if (!modelStr.includes('/') && this.baseURL !== "https://api.openai.com/v1") {
       // Construct a custom model object for pi-ai
       // We use 'openai-completions' API type for generic OpenAI compatibility
       return {
         id: modelId,
         name: modelId,
         api: 'openai-completions',
         provider: 'custom',
         baseUrl: this.baseURL,
         input: ['text'],
         contextWindow: 128000, 
         maxTokens: 4096,
         cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
       } as any; // Cast to any to avoid strict typing issues with Model union
    }

    try {
      // Try to get from registry
      return getModel(provider as any, modelId);
    } catch (e) {
      // Fallback for unknown models/providers - treat as custom OpenAI compatible
      return {
        id: modelId,
        name: modelId,
        api: 'openai-completions',
        provider: provider,
        baseUrl: this.baseURL,
        input: ['text'],
        contextWindow: 128000,
        maxTokens: 4096,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
      } as any;
    }
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    const modelStr = options.model || this.generateModel;
    const debugLines: string[] = [];
    debugLines.push(`[generate] model: ${modelStr} baseURL: ${this.baseURL}`);
    try {
      const model = this.resolvePiModel(modelStr);
      debugLines.push(`[generate] resolved → api: ${model.api}  provider: ${model.provider}  reasoning: ${(model as any).reasoning ?? '?'}  baseUrl: ${model.baseUrl}`);

      const context: Context = {
        messages: [{ role: 'user', content: prompt, timestamp: Date.now() }]
      };

      const response = await complete(model, context, {
        apiKey: this.apiKey,
        maxTokens: options.maxTokens,
        temperature: options.temperature
      });

      debugLines.push(`[generate] stopReason: ${response.stopReason}`);
      debugLines.push(`[generate] errorMessage: ${response.errorMessage ?? '(none)'}`);
      debugLines.push(`[generate] provider: ${response.provider}  api: ${response.api}  model: ${response.model}`);
      debugLines.push(`[generate] usage: ${JSON.stringify(response.usage)}`);
      debugLines.push(`[generate] content block types: ${response.content.map(b => b.type).join(', ') || '(none)'}`);

      // Extract text content — skip thinking blocks (reasoning model internal monologue)
      const text = response.content
        .filter((block): block is { type: 'text'; text: string } => block.type === 'text')
        .map(block => block.text)
        .join('');

      if (!text) {
        debugLines.push(`[generate] warning: empty text output`);
        process.stderr.write(debugLines.join('\n') + '\n');
      }

      return {
        text,
        model: modelStr,
        done: true
      };
    } catch (error) {
      process.stderr.write(debugLines.join('\n') + '\n');
      console.error("Remote generation error:", error);
      return null;
    }
  }

  async modelExists(model: string): Promise<ModelInfo> {
    // For remote, we assume configured models exist or let the API fail
    return {
      name: model,
      exists: true
    };
  }

  async expandQuery(query: string, options: { context?: string, includeLexical?: boolean } = {}): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;
    
    const prompt = `You are a search query expansion assistant.
Expand the following search query into multiple variations for different search backends.
Return ONLY a JSON array of objects with "type" and "text" fields.
Types can be:
- "lex": Lexical/keyword search (exact matches)
- "vec": Vector/semantic search (meaning)
- "hyde": Hypothetical Document Embeddings (answer to the query)

Query: ${query}

JSON Response:`;

    let result: Awaited<ReturnType<typeof this.generate>> = null;
    try {
      result = await this.generate(prompt, {
        temperature: 0.7,
        maxTokens: 600
      });

      if (!result) return [];

      let jsonStr = result.text.trim();

      // Remove markdown code blocks if present
      if (jsonStr.startsWith("```json")) {
        jsonStr = jsonStr.slice(7);
      }
      if (jsonStr.startsWith("```")) {
        jsonStr = jsonStr.slice(3);
      }
      if (jsonStr.endsWith("```")) {
        jsonStr = jsonStr.slice(0, -3);
      }
      jsonStr = jsonStr.trim();

      const parsed = JSON.parse(jsonStr);
      if (!Array.isArray(parsed)) return [];

      const queryables: Queryable[] = parsed
        .filter((item: any) =>
          item.type &&
          item.text &&
          ['lex', 'vec', 'hyde'].includes(item.type)
        )
        .map((item: any) => ({
          type: item.type as QueryType,
          text: item.text
        }));

      return includeLexical ? queryables : queryables.filter(q => q.type !== 'lex');
    } catch (error) {
      console.error(`Remote query expansion failed — prompt:\n${prompt}\n\nraw response:\n${result?.text ?? '(null)'}\n\nerror:`, error);
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return fallback;
    }
  }

  async rerank(query: string, documents: RerankDocument[], options: RerankOptions = {}): Promise<RerankResult> {
    // Most standard OpenAI-compatible APIs don't have a rerank endpoint.
    // Some providers like Cohere do, but it requires a different API structure.
    // For now, we'll log a warning and return documents in original order (or maybe simple Jaccard/keyword sort?)
    // Or we could use the LLM to rerank (listwise ranking), but that's complex and token-heavy.
    
    // If the user configured a specific rerank model that implies they have a provider that supports it,
    // we might try a specific endpoint. But for "generic" OpenAI compat, we can't assume.
    
    console.warn("Remote reranking is not fully implemented. Returning documents in original order.");
    
    return {
      results: documents.map((doc, index) => ({
        file: doc.file,
        score: 1.0 - (index * 0.001), // Dummy score preserving order
        index
      })),
      model: "noop-reranker"
    };
  }

  async getDeviceInfo(): Promise<{
    gpu: string | false;
    gpuOffloading: boolean;
    gpuDevices: string[];
    vram?: { total: number; used: number; free: number };
    cpuCores: number;
  }> {
    return {
      gpu: "cloud",
      gpuOffloading: false,
      gpuDevices: ["remote"],
      cpuCores: 0
    };
  }

  async tokenize(text: string): Promise<string[]> {
    // Simple regex to split by word boundaries but keep delimiters
    // This is an approximation for tokenization
    return text.match(/[\w]+|[^\w\s]|\s+/g) || [];
  }

  async detokenize(tokens: string[]): Promise<string> {
    return tokens.join('');
  }

  async dispose(): Promise<void> {
    // Nothing to dispose for remote
  }
}
