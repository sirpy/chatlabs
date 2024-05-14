import {
  BaseNode,
  VectorStore,
  VectorStoreQuery,
  VectorStoreQueryResult,
  MetadataFilters,
  getTopKEmbeddingsLearner,
  getTopKMMREmbeddings,
  VectorStoreQueryMode,
  getTopKEmbeddings,
  ExactMatchFilter,
  DEFAULT_PERSIST_DIR,
  exists
} from "llamaindex"

import type { GenericFileSystem } from "@llamaindex/env"
import { defaultFS, path } from "@llamaindex/env"
import _ from "lodash"

const LEARNER_MODES = new Set<VectorStoreQueryMode>([
  VectorStoreQueryMode.SVM,
  VectorStoreQueryMode.LINEAR_REGRESSION,
  VectorStoreQueryMode.LOGISTIC_REGRESSION
])

const MMR_MODE = VectorStoreQueryMode.MMR

class SimpleVectorStoreData {
  embeddingDict: Record<string, number[]> = {}
  textIdToRefDocId: Record<string, string> = {}
  metadataDict: Record<string, any> = {}
}

export interface InArrayFilter {
  filterType: "InArray"
  key: string
  value: (string | number)[]
}

function _buildMetadataFilterFn(
  metadataLookupFn: (nodeId: string) => Record<string, any>,
  metadataFilters: { filters: (InArrayFilter | ExactMatchFilter)[] }
): (nodeId: string) => boolean {
  return function filterFn(nodeId: string): boolean {
    const metadata = metadataLookupFn(nodeId)
    for (const filter of metadataFilters.filters) {
      const metadataValue = metadata[filter.key]
      if (filter.filterType === "InArray") {
        if (!filter.value.includes(metadataValue)) {
          return false
        }
      } else {
        // Exact match filter
        if (metadataValue === undefined || metadataValue !== filter.value) {
          return false
        }
      }
    }
    return true
  }
}

export class SimpleVectorStore2 implements VectorStore {
  storesText: boolean = false
  private data: SimpleVectorStoreData = new SimpleVectorStoreData()
  private fs: GenericFileSystem = defaultFS
  private persistPath: string | undefined

  constructor(data?: SimpleVectorStoreData, fs?: GenericFileSystem) {
    this.data = data || new SimpleVectorStoreData()
    this.fs = fs || defaultFS
  }

  static async fromPersistDir(
    persistDir: string = DEFAULT_PERSIST_DIR,
    fs: GenericFileSystem = defaultFS
  ): Promise<SimpleVectorStore2> {
    const persistPath = `${persistDir}/vector_store.json`
    return await SimpleVectorStore2.fromPersistPath(persistPath, fs)
  }

  get client(): any {
    return null
  }

  async get(textId: string): Promise<number[]> {
    return this.data.embeddingDict[textId]
  }

  async add(embeddingResults: BaseNode[]): Promise<string[]> {
    for (const node of embeddingResults) {
      this.data.embeddingDict[node.id_] = node.getEmbedding()

      if (!node.sourceNode) {
        continue
      }

      this.data.textIdToRefDocId[node.id_] = node.sourceNode?.nodeId
      this.data.metadataDict[node.id_] = node.metadata
    }

    if (this.persistPath) {
      await this.persist(this.persistPath, this.fs)
    }

    return embeddingResults.map(result => result.id_)
  }

  async delete(refDocId: string): Promise<void> {
    const textIdsToDelete = Object.keys(this.data.textIdToRefDocId).filter(
      textId => this.data.textIdToRefDocId[textId] === refDocId
    )
    for (const textId of textIdsToDelete) {
      delete this.data.embeddingDict[textId]
      delete this.data.textIdToRefDocId[textId]
      delete this.data.metadataDict[textId]
    }
    if (this.persistPath) {
      await this.persist(this.persistPath, this.fs)
    }
    return Promise.resolve()
  }

  async query(query: VectorStoreQuery): Promise<VectorStoreQueryResult> {
    let items = Object.entries(this.data.embeddingDict)
    console.log("items before filter:", items.length, query)
    console.log("metadata dict:", Object.entries(this.data.metadataDict).length)
    if (query.filters && query.filters.filters?.length > 0) {
      if (!this.data.metadataDict) {
        throw new Error("Metadata dictionary is not defined.")
      }

      const metadataFilterFn = _buildMetadataFilterFn(
        nodeId => this.data.metadataDict[nodeId],
        { filters: query.filters.filters }
      )

      items = items.filter(([nodeId]) => metadataFilterFn(nodeId))
    }

    console.log("items after filter:", items.length)
    let nodeIds: string[], embeddings: number[][]
    if (query.docIds && query.docIds.length > 0) {
      let availableIds = new Set(query.docIds)
      const queriedItems = items.filter(item => availableIds.has(item[0]))
      nodeIds = queriedItems.map(item => item[0])
      embeddings = queriedItems.map(item => item[1])
    } else {
      nodeIds = items.map(item => item[0])
      embeddings = items.map(item => item[1])
    }

    let queryEmbedding = query.queryEmbedding!

    let topSimilarities: number[], topIds: string[]
    if (LEARNER_MODES.has(query.mode)) {
      ;[topSimilarities, topIds] = getTopKEmbeddingsLearner(
        queryEmbedding,
        embeddings,
        query.similarityTopK,
        nodeIds
      )
    } else if (query.mode === MMR_MODE) {
      let mmrThreshold = query.mmrThreshold
      ;[topSimilarities, topIds] = getTopKMMREmbeddings(
        queryEmbedding,
        embeddings,
        null,
        query.similarityTopK,
        nodeIds,
        mmrThreshold
      )
    } else if (query.mode === VectorStoreQueryMode.DEFAULT) {
      ;[topSimilarities, topIds] = getTopKEmbeddings(
        queryEmbedding,
        embeddings,
        query.similarityTopK,
        nodeIds
      )
    } else {
      throw new Error(`Invalid query mode: ${query.mode}`)
    }

    return {
      similarities: topSimilarities,
      ids: topIds
    }
  }

  async persist(
    persistPath: string = `${DEFAULT_PERSIST_DIR}/vector_store.json`,
    fs?: GenericFileSystem
  ): Promise<void> {
    fs = fs || this.fs
    const dirPath = path.dirname(persistPath)
    if (!(await exists(fs, dirPath))) {
      await fs.mkdir(dirPath)
    }

    await fs.writeFile(persistPath, JSON.stringify(this.data))
  }

  static async fromPersistPath(
    persistPath: string,
    fs: GenericFileSystem = defaultFS
  ): Promise<SimpleVectorStore2> {
    const dirPath = path.dirname(persistPath)
    if (!(await exists(fs, dirPath))) {
      await fs.mkdir(dirPath, { recursive: true })
    }

    let dataDict: any = {}
    try {
      const fileData = await fs.readFile(persistPath)
      dataDict = JSON.parse(fileData.toString())
    } catch (e) {
      console.error(
        `No valid data found at path: ${persistPath} starting new store.`
      )
    }

    const data = new SimpleVectorStoreData()
    data.embeddingDict = dataDict.embeddingDict ?? {}
    data.textIdToRefDocId = dataDict.textIdToRefDocId ?? {}
    data.metadataDict = dataDict.metadataDict ?? {}
    const store = new SimpleVectorStore2(data)
    store.persistPath = persistPath
    store.fs = fs
    return store
  }

  static fromDict(saveDict: SimpleVectorStoreData): SimpleVectorStore2 {
    const data = new SimpleVectorStoreData()
    data.embeddingDict = saveDict.embeddingDict
    data.textIdToRefDocId = saveDict.textIdToRefDocId
    data.metadataDict = saveDict.metadataDict
    return new SimpleVectorStore2(data)
  }

  toDict(): SimpleVectorStoreData {
    return {
      embeddingDict: this.data.embeddingDict,
      textIdToRefDocId: this.data.textIdToRefDocId,
      metadataDict: this.data.metadataDict
    }
  }
}
