import { generateLocalEmbedding } from "@/lib/generate-local-embedding"
import { checkApiKey, getServerProfile } from "@/lib/server/server-chat-helpers"
import { Database } from "@/supabase/types"
import { createClient } from "@supabase/supabase-js"
import OpenAI from "openai"
import { groupBy, uniq } from "lodash"

export async function POST(request: Request) {
  const json = await request.json()
  const { userInput, fileIds, embeddingsProvider, sourceCount } = json as {
    userInput: string
    fileIds: string[]
    embeddingsProvider: "openai" | "local"
    sourceCount: number
  }

  const uniqueFileIds = [...new Set(fileIds)]

  try {
    const supabaseAdmin = createClient<Database>(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    )

    const profile = await getServerProfile()

    if (embeddingsProvider === "openai") {
      if (profile.use_azure_openai) {
        checkApiKey(profile.azure_openai_api_key, "Azure OpenAI")
      } else {
        checkApiKey(profile.openai_api_key, "OpenAI")
      }
    }

    let chunks: any[] = []

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

    if (embeddingsProvider === "openai") {
      const response = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: userInput
      })

      const openaiEmbedding = response.data.map(item => item.embedding)[0]

      const { data: openaiFileItems, error: openaiError } =
        await supabaseAdmin.rpc("match_file_items_openai", {
          query_embedding: openaiEmbedding as any,
          match_count: sourceCount,
          file_ids: uniqueFileIds
        })

      if (openaiError) {
        throw openaiError
      }

      chunks = openaiFileItems
    } else if (embeddingsProvider === "local") {
      console.log("retrieve", { userInput })
      const localEmbedding = await generateLocalEmbedding(userInput)

      // console.log("localEmbedding", localEmbedding, sourceCount, uniqueFileIds)
      // const { index, storageContext } = await getDatabase()

      // const retriever = await index.asRetriever({ similarityTopK: 3 })
      // const mdFilters = [
      //   {
      //     key: "file_id",
      //     value: fileIds,
      //     filterType: "InArray",
      //   },
      // ]
      // const results = await retriever.retrieve({ query: userInput, preFilters: { filters: mdFilters } })
      // console.log("node results:", results)
      // const best = results[0].score || 0
      // const pages = []
      // for (let node of results) {
      //   if (node.score && node.score < best * 0.995) continue;
      //   if (node.node.sourceNode) {
      //     const doc = await index.docStore.getDocument(node.node.sourceNode?.nodeId, false)
      //     // console.log("found chunk doc metadata:", doc?.metadata, node.node.metadata, node.node.relationships)
      //     pages.push({ id: doc?.id_, file_id: doc?.metadata?.file_id, content: doc?.getContent(MetadataMode.NONE), similarity: node.score })
      //   }
      // }
      // chunks = uniqBy(pages, 'id')
      // console.log("returning pages:", chunks.length, chunks)

      // const { data: localFileItems, error: localFileItemsError } =
      //   await supabaseAdmin.rpc("match_file_items_local", {
      //     query_embedding: localEmbedding as any,
      //     match_count: sourceCount,
      //     file_ids: uniqueFileIds
      //   })

      const { data: localPageItems, error: localPageItemsError } =
        await supabaseAdmin.rpc("match_file_pages_local", {
          query_embedding: localEmbedding as any,
          match_count: sourceCount,
          file_ids: uniqueFileIds
        })
      if (localPageItemsError) {
        throw localPageItemsError
      }

      const groups = groupBy(localPageItems, _ =>
        _.similarity === -1 ? "pages" : "items"
      )
      const pages = groups["pages"]
      const items = groups["items"]
      const best = items[0]
      const validPages = uniq(
        items
          .filter(_ => _.similarity >= best.similarity * 0.995)
          .map(_ => _.source)
      )
      chunks = pages.filter(_ => validPages.includes(_.id))

      // chunks = localFileItems
      console.log("found db items", {
        localPageItems,
        localPageItemsError,
        validPages
      })
      console.log("returning pages:", chunks.length, "best:", best.similarity)
    }

    // const mostSimilarChunks = chunks?.sort(
    //   (a, b) => b.similarity - a.similarity
    // )

    return new Response(JSON.stringify({ results: chunks }), {
      status: 200
    })
  } catch (error: any) {
    console.error(error)
    const errorMessage = error.error?.message || "An unexpected error occurred"
    const errorCode = error.status || 500
    return new Response(JSON.stringify({ message: errorMessage }), {
      status: errorCode
    })
  }
}
