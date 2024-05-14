import { pipeline } from "@xenova/transformers"

export async function batchEmbeddings(
  values: string[],
  chunkSize: number,
  options: any
) {
  const getTextEmbeddings = (texts: string[]) => {
    const embeddings = []
    for (const text of texts) {
      const embedding = generateLocalEmbedding(text)
      embeddings.push(embedding)
    }
    return Promise.all(embeddings)
  }
  const resultEmbeddings = []
  const queue = values
  const curBatch = []
  for (let i = 0; i < queue.length; i++) {
    curBatch.push(queue[i])
    if (i == queue.length - 1 || curBatch.length == chunkSize) {
      const embeddings = await getTextEmbeddings(curBatch)
      resultEmbeddings.push(...embeddings)
      if (options?.logProgress) {
        console.log(`getting embedding progress: ${i} / ${queue.length}`)
      }
      curBatch.length = 0
    }
  }
  return resultEmbeddings
}

const generateEmbeddingPromise = pipeline(
  "feature-extraction",
  "Xenova/multilingual-e5-small"
)

export async function generateLocalEmbedding(content: string) {
  const generateEmbedding = await generateEmbeddingPromise
  console.log("Generated local embedding generating....")
  const output = await generateEmbedding(content, {
    pooling: "mean",
    normalize: true
  })

  console.log("Generated local embedding generating.... done")
  const embedding = Array.from(output.data)

  return embedding
}
