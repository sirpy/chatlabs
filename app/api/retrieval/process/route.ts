import {
  processCSV,
  processJSON,
  processMarkdown,
  processPdf,
  processTxt
} from "@/lib/retrieval/processing"
import { splitPagesToChunks } from "@/lib/llamaindex/database"
import { checkApiKey, getServerProfile } from "@/lib/server/server-chat-helpers"
import { Database } from "@/supabase/types"
import { FileItemChunk } from "@/types"
import { createClient } from "@supabase/supabase-js"
import { NextResponse } from "next/server"
import OpenAI from "openai"
import { Document, MetadataMode, RelatedNodeType } from "llamaindex"
import { createSHA256 } from "@llamaindex/env"
import { batchEmbeddings } from "@/lib/generate-local-embedding"
import { Metadata } from "next"

const maxDuration = 300
export async function POST(req: Request) {
  try {
    console.log(
      "process request",
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    )
    const supabaseAdmin = createClient<Database>(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    )

    const profile = await getServerProfile()

    console.log("process request", profile)
    const formData = await req.formData()

    const file = formData.get("file") as File
    const file_id = formData.get("file_id") as string
    const embeddingsProvider = formData.get("embeddingsProvider") as string

    console.log("process request", { file_id, embeddingsProvider })

    const fileBuffer = Buffer.from(await file.arrayBuffer())
    const blob = new Blob([fileBuffer])
    const fileExtension = file.name.split(".").pop()?.toLowerCase()

    if (embeddingsProvider === "openai") {
      if (profile.use_azure_openai) {
        checkApiKey(profile.azure_openai_api_key, "Azure OpenAI")
      } else {
        checkApiKey(profile.openai_api_key, "OpenAI")
      }
    }

    let chunks: FileItemChunk[] = []
    console.log("process request", { fileExtension })
    switch (fileExtension) {
      case "csv":
        chunks = await processCSV(blob)
        break
      case "json":
        chunks = await processJSON(blob)
        break
      case "md":
        chunks = await processMarkdown(blob)
        break
      case "pdf":
        chunks = await processPdf(blob)
        break
      case "txt":
        chunks = await processTxt(blob)
        break
      default:
        return new NextResponse("Unsupported file type", {
          status: 400
        })
    }

    let embeddings: any = []

    let openai
    if (profile.use_azure_openai) {
      openai = new OpenAI({
        apiKey: profile.azure_openai_api_key || "",
        baseURL: `${profile.azure_openai_endpoint}/openai/deployments/${profile.azure_openai_embeddings_id}`,
        defaultQuery: { "api-version": "2023-12-01-preview" },
        defaultHeaders: { "api-key": profile.azure_openai_api_key }
      })
    } else {
      openai = new OpenAI({
        apiKey: profile.openai_api_key || "",
        organization: profile.openai_organization_id
      })
    }

    // const { indexDocuments } = await getDatabase()

    console.log(
      "docsMeta sample",
      chunks.slice(0, 2).map(_ => _.metadata)
    )
    const sha256 = createSHA256()
    chunks.forEach(_ => sha256.update(_.content))
    const deterministic_file_id = sha256.digest()
    let docs = chunks.map(
      _ =>
        new Document({
          text: _.content,
          metadata: { loc: _.metadata?.loc, file_id, deterministic_file_id }
        })
    )
    console.log("smaple doc metadata:", docs[0].metadata)
    const nodes = await splitPagesToChunks(docs)
    console.log("nodes:", nodes.slice(0, 5), " starting embeddings...")
    // await indexDocuments(docs)

    if (embeddingsProvider === "openai") {
      const response = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: chunks.map(chunk => chunk.content)
      })

      embeddings = response.data.map((item: any) => {
        return item.embedding
      })
    } else if (embeddingsProvider === "local") {
      embeddings = await batchEmbeddings(
        nodes.map(_ => _.getContent(MetadataMode.NONE)),
        Number(process.env.EMBEDDING_BATCH_SIZE) || 100,
        { logProgress: true }
      )
    }

    console.log("process request", "got embeddings", embeddings.length)

    const file_items = nodes.map((chunk, index) => ({
      id: chunk.id_,
      file_id,
      source: (chunk.relationships.SOURCE as any).nodeId,
      next: (chunk.relationships.NEXT as any).nodeId,
      // metadata: chunk.metadata,
      user_id: profile.user_id,
      content: chunk.getContent(MetadataMode.NONE),
      tokens: chunk.getContent(MetadataMode.NONE).length,
      openai_embedding:
        embeddingsProvider === "openai"
          ? ((embeddings[index] || null) as any)
          : null,
      local_embedding:
        embeddingsProvider === "local"
          ? ((embeddings[index] || null) as any)
          : null
    }))

    const pages = docs.map((chunk, index) => ({
      id: chunk.id_,
      file_id,
      source: (chunk.relationships.SOURCE as any).nodeId,
      next: (chunk.relationships.NEXT as any).nodeId,
      // metadata: chunk.metadata,
      user_id: profile.user_id,
      content: chunk.getContent(MetadataMode.NONE),
      tokens: chunk.getContent(MetadataMode.NONE).length,
      openai_embedding: null,
      local_embedding: null
    }))

    console.log(
      "process request",
      "inserting to supabase",
      "chunks:",
      file_items.length,
      "pages:",
      pages.length
    )

    const upsertResult = await supabaseAdmin
      .from("file_items")
      .upsert(file_items.concat(pages))

    const totalTokens = file_items.reduce((acc, item) => acc + item.tokens, 0)
    console.log("process request", "inserted to supabase", {
      totalTokens,
      upsertResult
    })

    await supabaseAdmin
      .from("files")
      .update({ tokens: totalTokens })
      .eq("id", file_id)

    return new Response(
      JSON.stringify({ message: "Embed Successful", deterministic_file_id }),
      {
        status: 200
      }
    )
  } catch (error: any) {
    console.log(error)
    const errorMessage = error.error?.message || "An unexpected error occurred"
    const errorCode = error.status || 500
    return new Response(JSON.stringify({ message: errorMessage }), {
      status: errorCode
    })
  }
}
