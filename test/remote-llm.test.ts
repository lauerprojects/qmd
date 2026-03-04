import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { RemoteLLM } from "../src/remote-llm";
import * as piAi from "@mariozechner/pi-ai";

// Mock the entire module
vi.mock("@mariozechner/pi-ai", async () => {
  return {
    getModel: vi.fn(),
    complete: vi.fn(),
  };
});

describe("RemoteLLM", () => {
  const config = {
    apiKey: "test-key",
    baseURL: "https://api.example.com/v1"
  };

  let llm: RemoteLLM;

  beforeEach(() => {
    llm = new RemoteLLM(config);
    global.fetch = vi.fn();
    vi.resetAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should initialize with correct config", () => {
    expect(llm).toBeInstanceOf(RemoteLLM);
  });

  it("should call embeddings endpoint via fetch", async () => {
    const mockResponse = {
      data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }]
    };
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    });

    const result = await llm.embed("test");
    expect(result).toEqual({
      embedding: [0.1, 0.2, 0.3],
      model: "text-embedding-3-small" // default
    });

    expect(global.fetch).toHaveBeenCalledWith(
      "https://api.example.com/v1/embeddings",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "Authorization": "Bearer test-key"
        }),
        body: JSON.stringify({
          model: "text-embedding-3-small",
          input: "test"
        })
      })
    );
  });

  it("should call pi-ai complete for generate", async () => {
    const mockModel = { id: "gpt-3.5-turbo", api: "openai" };
    (piAi.getModel as any).mockReturnValue(mockModel);
    
    (piAi.complete as any).mockResolvedValue({
      content: [{ type: "text", text: "Hello world" }]
    });

    const result = await llm.generate("Hi");
    
    expect(piAi.getModel).toHaveBeenCalled();
    expect(piAi.complete).toHaveBeenCalledWith(
      expect.objectContaining({ id: "gpt-3.5-turbo" }),
      expect.objectContaining({
        messages: [{ role: "user", content: "Hi" }]
      }),
      expect.objectContaining({
        apiKey: "test-key"
      })
    );

    expect(result).toEqual({
      text: "Hello world",
      model: "openai/gpt-3.5-turbo", // default
      done: true
    });
  });
});
