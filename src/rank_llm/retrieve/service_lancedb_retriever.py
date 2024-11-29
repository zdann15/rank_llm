from urllib import parse

import requests

from rank_llm.data import Candidate, Query, Request

from . import RetrievalMethod, RetrievalMode


class ServiceLanceDBRetriever:
    def __init__(
        self,
        retrieval_mode: RetrievalMode = RetrievalMode.DATASET,
        retrieval_method: RetrievalMethod = RetrievalMethod.LANCE_ARCTIC_EMBED_L,
    ) -> None:
        """
        Creates a ServiceLanceDBRetriever instance with a specified retrieval method and mode.

        Args:
            retrieval_mode (RetrievalMode): The retrieval mode to be used. Defaults to DATASET. Only DATASET mode is currently supported.
            retrieval_method (RetrievalMethod): The retrieval method to be used. Defaults to BM25.

        Raises:
            ValueError: If retrieval mode or retrieval method is invalid or missing.
        """
        self._retrieval_mode = retrieval_mode
        self._retrieval_method = retrieval_method

        if retrieval_mode != RetrievalMode.DATASET:
            raise ValueError(
                f"{retrieval_mode} is not supported for ServiceRetriever. Only DATASET mode is currently supported."
            )
        print(retrieval_method)
        print(RetrievalMethod.LANCE_FTS)
        if retrieval_method not in [
            RetrievalMethod.LANCE_FTS,
            RetrievalMethod.LANCE_ARCTIC_EMBED_L,
            RetrievalMethod.LANCE_HYBRID,
        ]:
            if retrieval_method == "lance_fts":
                pass
            else:
                raise ValueError(
                    f"{retrieval_method} is not supported for ServiceLanceDBRetriever. Only LANCE_FTS, LANCE_ARCTIC_EMBED_L, LANCE_HYBRID are currently supported."
                )

    def retrieve(
        self,
        dataset: str,
        request: Request,
        k: int = 50,
        host: str = "http://localhost:8081",
        timeout: int = 15
        * 60,  # downloding and decompressing the index can take a long time.
    ) -> Request:
        """
        Executes the retrieval process based on the configation provided with the Retriever instance. Takes in a Request object with a query and empty candidates object and the top k items to retrieve.

        Args:
            request (Request): The request containing the query and qid.
            dataset (str): The name of the dataset.
            k (int, optional): The top k hits to retrieve. Defaults to 100.
            host (str): The Anserini API host address. Defaults to http://localhost:8081

        Returns:
            Request. Contains a query and list of candidates
        Raises:
            ValueError: If the retrieval mode is invalid or the result format is not as expected.
        """
        # http://localhost:8042/api/v1.0/indexes/msmarco-v2.1-doc-segmented/hybrid
        retrieval_method = str(self._retrieval_method)
        print(retrieval_method)
        if retrieval_method == "lance_fts":
            retrieval_method = "fts"
        elif retrieval_method == "lance_arctic_embed_l":
            retrieval_method = "vector"
        else:
            retrieval_method = "hybrid"
        url = f"{host}/api/v1.0/indexes/{dataset}/{retrieval_method}?query={parse.quote(request.query.text)}&hits={str(k)}&qid={request.query.qid}"
        print(url)
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise type(e)(
                f"Failed to retrieve data from LanceDB server: {str(e)}"
            ) from e

        data = response.json()
        retrieved_results = Request(
            query=Query(text=data["query"]["text"], qid=data["query"]["qid"])
        )

        for candidate in data["candidates"]:
            retrieved_results.candidates.append(
                Candidate(
                    docid=candidate["docid"],
                    score=candidate["score"],
                    doc=candidate["doc"],
                )
            )

        return retrieved_results
