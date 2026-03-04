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
  TokenLogProb
} from "./llm-types.js";
import { RemoteLLM } from "./remote-llm.js";

export type LLMBackend = 'local' | 'remote';

export type HybridLLMConfig = {
  embedBackend: LLMBackend;
  generateBackend: LLMBackend;
  rerankBackend: LLMBackend;
  tokenizeBackend: LLMBackend;
};

export class HybridLLM implements LLM {
  private local: LLM;
  private remote?: RemoteLLM;
  private config: HybridLLMConfig;

  constructor(local: LLM, remote: RemoteLLM | undefined, config: HybridLLMConfig) {
    this.local = local;
    this.remote = remote;
    this.config = config;
  }

  private getBackend(preference: LLMBackend): LLM {
    if (preference === 'remote') {
      if (!this.remote) {
        console.warn("Remote backend requested but not available (no API key). Falling back to local.");
        return this.local;
      }
      return this.remote;
    }
    return this.local;
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    return this.getBackend(this.config.embedBackend).embed(text, options);
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return this.getBackend(this.config.embedBackend).embedBatch(texts);
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    return this.getBackend(this.config.generateBackend).generate(prompt, options);
  }

  async modelExists(model: string): Promise<ModelInfo> {
    // Check both? Or just the one that matches the backend for that model type?
    // Since we don't know what type 'model' is (embed/gen/rerank), we might need to check both.
    // But usually this is used to check if a specific model file exists locally.
    const localInfo = await this.local.modelExists(model);
    if (localInfo.exists) return localInfo;

    if (this.remote) {
      return this.remote.modelExists(model);
    }

    return localInfo;
  }

  async expandQuery(query: string, options: { context?: string, includeLexical?: boolean } = {}): Promise<Queryable[]> {
    // Usually tied to generation
    return this.getBackend(this.config.generateBackend).expandQuery(query, options);
  }

  async rerank(query: string, documents: RerankDocument[], options: RerankOptions = {}): Promise<RerankResult> {
    return this.getBackend(this.config.rerankBackend).rerank(query, documents, options);
  }

  async tokenize(text: string): Promise<readonly any[]> {
    return this.getBackend(this.config.tokenizeBackend).tokenize(text);
  }

  async detokenize(tokens: readonly any[]): Promise<string> {
    return this.getBackend(this.config.tokenizeBackend).detokenize(tokens);
  }

  async getDeviceInfo(): Promise<{
    gpu: string | false;
    gpuOffloading: boolean;
    gpuDevices: string[];
    vram?: { total: number; used: number; free: number };
    cpuCores: number;
  }> {
    // Combine info? Or just return local info since that's where hardware matters?
    // If we are fully remote, local hardware info is irrelevant but harmless.
    // If we use local for anything (like rerank), we want to see GPU status.
    return this.local.getDeviceInfo();
  }

  async dispose(): Promise<void> {
    await this.local.dispose();
    if (this.remote) {
      await this.remote.dispose();
    }
  }
}
