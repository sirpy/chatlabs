import { once } from "lodash"
import { CHUNK_OVERLAP, CHUNK_SIZE } from "../retrieval/processing"
import {
  Settings,
  storageContextFromDefaults,
  VectorStoreIndex,
  HuggingFaceEmbedding,
  Document,
  runTransformations,
  SimpleNodeParser,
  Metadata
} from "llamaindex"
import {
  DocStoreStrategy,
  createDocStoreStrategy
} from "llamaindex/ingestion/strategies/index"
import { nodeParserFromSettingsOrContext } from "llamaindex/Settings"
import { defaultFS } from "@llamaindex/env"
import { env } from "@xenova/transformers"
import { SimpleVectorStore2 } from "./SimpleVectorStore2"

env.cacheDir = "./xenovalm-cache"
const embedModel = new HuggingFaceEmbedding({
  modelType: "Xenova/multilingual-e5-small"
})
embedModel.embedBatchSize = 100
Settings.embedModel = embedModel
Settings.chunkOverlap = CHUNK_OVERLAP
Settings.chunkSize = CHUNK_SIZE

const nodeParser = new SimpleNodeParser({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP
})

export const splitPagesToChunks = (pages: Document<Metadata>[]) => {
  return nodeParser.transform(pages)
}

export const getDatabase = once(async () => {
  console.log("Get Database called")
  const storageContext = await storageContextFromDefaults({
    persistDir: `./llamaindex`,
    vectorStore: await SimpleVectorStore2.fromPersistDir(
      "./llamaindex",
      defaultFS
    )
  })
  let index: VectorStoreIndex
  try {
    index = await VectorStoreIndex.init({ storageContext })
  } catch (e) {
    console.log("unable to open vector index. creating new...")
    index = await VectorStoreIndex.init({
      nodes: [],
      storageContext
    })
  }

  console.log(
    "Database initialized",
    Object.keys(
      (index.vectorStore as SimpleVectorStore2).toDict().embeddingDict
    ).length
  )
  const indexDocuments = async (docs: Document[]) => {
    console.log("llama index. to insert:", docs.length)
    // use doc store strategy to avoid duplicates
    const docStoreStrategy = createDocStoreStrategy(
      DocStoreStrategy.DUPLICATES_ONLY,
      storageContext.docStore,
      storageContext.vectorStore
    )
    const nodes = await runTransformations(
      docs,
      [nodeParserFromSettingsOrContext(undefined)],
      {},
      {
        docStoreStrategy
      }
    )
    await index.insertNodes(nodes)
  }

  return {
    storageContext,
    index,
    indexDocuments
  }
})
