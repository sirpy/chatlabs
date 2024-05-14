import { FileItemChunk } from "@/types"
import { encode } from "gpt-tokenizer"
import { PDFLoader } from "langchain/document_loaders/fs/pdf"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { CHUNK_OVERLAP, CHUNK_SIZE } from "."

export const processPdf = async (pdf: Blob): Promise<FileItemChunk[]> => {
  const loader = new PDFLoader(pdf)
  const pages = await loader.load()
  // let completeText = docs.map(doc => doc.pageContent).join(" ")

  // const splitter = new RecursiveCharacterTextSplitter({
  //   chunkSize: CHUNK_SIZE,
  //   chunkOverlap: CHUNK_OVERLAP

  // })
  // console.log("processing pdf:",docs.length)
  // const splitDocs = await splitter.splitDocuments(docs)

  // console.log("splitDocs:",splitDocs.length, splitDocs[0], splitDocs[50])
  // let chunks: FileItemChunk[] = []

  const chunks: FileItemChunk[] = pages.map(page => {
    return {
      content: page.pageContent,
      metadata: page.metadata,
      tokens: encode(page.pageContent).length
    }
  })

  return chunks
}
