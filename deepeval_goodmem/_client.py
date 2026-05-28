"""HTTP client for the GoodMem API."""

import base64
import json
import mimetypes
import time
from typing import Any

import httpx


class GoodMemClient:
    """Low-level HTTP client for communicating with the GoodMem API.

    Handles authentication, URL normalization, and common request patterns
    used by all GoodMem operations.

    Args:
        base_url: The base URL of the GoodMem API server.
        api_key: The API key for authentication via `X-API-Key` header.
        timeout: Request timeout in seconds.
        verify_ssl: Whether to verify SSL certificates.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        verify_ssl: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._verify_ssl = verify_ssl

    @property
    def base_url(self) -> str:
        """The normalized base URL."""
        return self._base_url

    def _headers(self, *, content_type: str = "application/json") -> dict[str, str]:
        """Build common request headers.

        Args:
            content_type: The Content-Type header value.

        Returns:
            Dictionary of HTTP headers.
        """
        return {
            "X-API-Key": self._api_key,
            "Content-Type": content_type,
            "Accept": "application/json",
        }

    def _ndjson_headers(self) -> dict[str, str]:
        """Build headers for NDJSON streaming requests.

        Returns:
            Dictionary of HTTP headers with NDJSON accept type.
        """
        return {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        }

    def _url(self, path: str) -> str:
        """Build a full URL from a path.

        Args:
            path: The API path, e.g. `/v1/spaces`.

        Returns:
            The full URL.
        """
        return f"{self._base_url}{path}"

    def _client(self) -> httpx.Client:
        """Create a new synchronous HTTP client.

        Returns:
            A configured `httpx.Client` instance.
        """
        return httpx.Client(timeout=self._timeout, verify=self._verify_ssl)

    # -- Space operations --

    _DEFAULT_CHUNK_SIZE: int = 512
    _DEFAULT_CHUNK_OVERLAP: int = 50

    def create_space(
        self,
        *,
        name: str,
        embedder_id: str,
        chunking_strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> dict[str, Any]:
        """Create a new space or return an existing one with the same name.

        First lists existing spaces to check for a name match. If found,
        returns the existing space info. Otherwise creates a new space.

        Args:
            name: The name of the space.
            embedder_id: The ID of the embedder to associate with the space.
            chunking_strategy: The chunking strategy for the space.
                One of `recursive`, `sentence`, or `none`.
            chunk_size: The maximum chunk size in characters for recursive or
                sentence strategies.
            chunk_overlap: The overlap between consecutive chunks in characters.

        Returns:
            A dictionary with space info and a `reused` flag.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        try:
            spaces = self.list_spaces()
            for space in spaces:
                if space.get("name") == name:
                    return {
                        "success": True,
                        "spaceId": space["spaceId"],
                        "name": space["name"],
                        "embedderId": embedder_id,
                        "message": "Space already exists, reusing existing space",
                        "reused": True,
                    }
        except httpx.HTTPStatusError:
            pass

        if chunking_strategy == "none":
            chunking_config: dict[str, Any] = {"none": {}}
        else:
            chunking_config = {
                chunking_strategy: {
                    "chunkSize": chunk_size,
                    "chunkOverlap": chunk_overlap,
                },
            }

        with self._client() as client:
            response = client.post(
                self._url("/v1/spaces"),
                headers=self._headers(),
                json={
                    "name": name,
                    "spaceEmbedders": [{"embedderId": embedder_id}],
                    "defaultChunkingConfig": chunking_config,
                },
            )
            response.raise_for_status()
            body = response.json()

        return {
            "success": True,
            "spaceId": body["spaceId"],
            "name": body["name"],
            "embedderId": embedder_id,
            "message": "Space created successfully",
            "reused": False,
        }

    def list_spaces(self) -> list[dict[str, Any]]:
        """List all spaces.

        Returns:
            A list of space dictionaries.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        with self._client() as client:
            response = client.get(
                self._url("/v1/spaces"),
                headers=self._headers(),
            )
            response.raise_for_status()
            body = response.json()

        if isinstance(body, list):
            return body
        return body.get("spaces", [])

    def get_space(self, *, space_id: str) -> dict[str, Any]:
        """Fetch a space by ID.

        Args:
            space_id: The UUID of the space.

        Returns:
            The space dictionary.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        with self._client() as client:
            response = client.get(
                self._url(f"/v1/spaces/{space_id}"),
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()

    def update_space(
        self,
        *,
        space_id: str,
        name: str | None = None,
        public_read: bool | None = None,
        replace_labels: dict[str, str] | None = None,
        merge_labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Update mutable fields on a space.

        Only `name`, `publicRead`, and labels are mutable. Fields not provided
        are left unchanged.

        Args:
            space_id: The UUID of the space to update.
            name: New name for the space.
            public_read: Whether the space should be public-read.
            replace_labels: If provided, replaces the entire label set.
            merge_labels: If provided, merges into the existing label set.

        Returns:
            The updated space dictionary.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if public_read is not None:
            body["publicRead"] = public_read
        if replace_labels is not None:
            body["replaceLabels"] = replace_labels
        if merge_labels is not None:
            body["mergeLabels"] = merge_labels

        with self._client() as client:
            response = client.put(
                self._url(f"/v1/spaces/{space_id}"),
                headers=self._headers(),
                json=body,
            )
            response.raise_for_status()
            return response.json()

    def delete_space(self, *, space_id: str) -> dict[str, Any]:
        """Delete a space by ID.

        Args:
            space_id: The UUID of the space to delete.

        Returns:
            A dictionary confirming the deletion.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        with self._client() as client:
            response = client.delete(
                self._url(f"/v1/spaces/{space_id}"),
                headers=self._headers(),
            )
            response.raise_for_status()

        return {
            "success": True,
            "spaceId": space_id,
            "message": "Space deleted successfully",
        }

    # -- Memory operations --

    def create_memory(
        self,
        *,
        space_id: str,
        text_content: str | None = None,
        file_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new memory in a space from text or a file.

        If both `file_path` and `text_content` are provided, the file takes
        priority. The file content type is auto-detected from its extension.

        Args:
            space_id: The UUID of the target space.
            text_content: Plain text content to store.
            file_path: Local file path to upload as memory.
            metadata: Optional key-value metadata as a dictionary.

        Returns:
            A dictionary with the created memory info.

        Raises:
            ValueError: If neither `text_content` nor `file_path` is provided.
            httpx.HTTPStatusError: If the API returns an error status.
        """
        request_body: dict[str, Any] = {"spaceId": space_id}

        if file_path:
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                mime_type = "application/octet-stream"

            with open(file_path, "rb") as f:
                file_bytes = f.read()

            if mime_type.startswith("text/"):
                request_body["contentType"] = mime_type
                request_body["originalContent"] = file_bytes.decode("utf-8")
            else:
                request_body["contentType"] = mime_type
                request_body["originalContentB64"] = base64.b64encode(file_bytes).decode("ascii")
        elif text_content is not None:
            request_body["contentType"] = "text/plain"
            request_body["originalContent"] = text_content
        else:
            msg = "No content provided. Provide either text_content or file_path."
            raise ValueError(msg)

        if metadata:
            request_body["metadata"] = metadata

        with self._client() as client:
            response = client.post(
                self._url("/v1/memories"),
                headers=self._headers(),
                json=request_body,
            )
            response.raise_for_status()
            body = response.json()

        return {
            "success": True,
            "memoryId": body["memoryId"],
            "spaceId": body["spaceId"],
            "status": body.get("processingStatus", "PENDING"),
            "contentType": request_body["contentType"],
            "message": "Memory created successfully",
        }

    _CHAT_POSTPROCESSOR = "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory"

    def retrieve_memories(
        self,
        *,
        query: str,
        space_ids: str,
        max_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
        max_wait_seconds: float = 60.0,
        poll_interval: float = 5.0,
        reranker_id: str | None = None,
        llm_id: str | None = None,
        relevance_threshold: float | None = None,
        llm_temperature: float | None = None,
        chronological_resort: bool | None = None,
        metadata_filter: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve memories via semantic similarity search.

        Polls for results until `max_wait_seconds` elapses when
        `wait_for_indexing` is enabled and no results are found initially.

        When any of `reranker_id`, `llm_id`, `relevance_threshold`,
        `llm_temperature`, or `chronological_resort` is provided, a
        `ChatPostProcessor` stage is appended that reranks, filters,
        re-sorts, and/or generates an LLM summary (`abstractReply`).

        Args:
            query: The natural language search query.
            space_ids: Comma-separated space UUIDs to search across.
            max_results: Maximum number of matching chunks to return. Also
                used as the post-processor `max_results` when one is configured.
            include_memory_definition: Whether to include full memory metadata.
            wait_for_indexing: Whether to poll for results when none are
                returned on the first call.
            max_wait_seconds: Maximum time in seconds to poll for results when
                `wait_for_indexing` is enabled.
            poll_interval: Seconds to sleep between polling attempts when
                `wait_for_indexing` is enabled.
            reranker_id: UUID of a reranker to improve result ordering.
            llm_id: UUID of an LLM that generates a contextual `abstractReply`.
            relevance_threshold: Minimum relevance score (0-1) for inclusion.
            llm_temperature: Creativity setting for the LLM (0-2).
            chronological_resort: Reorder final results by creation time.
            metadata_filter: SQL-style JSONPath expression applied server-side
                to narrow results by metadata. When set, the same filter is
                attached to every space key in the request. Empty strings are
                ignored. Example: `CAST(val('$.category') AS TEXT) = 'feat'`.

        Returns:
            A dictionary containing matched results and memory definitions.
            Includes `abstractReply` when an LLM summary was generated.

        Raises:
            ValueError: If no valid space IDs are provided.
            httpx.HTTPStatusError: If the API returns an error status.
        """
        space_keys: list[dict[str, Any]] = [
            {"spaceId": sid.strip()} for sid in space_ids.split(",") if sid.strip()
        ]
        if not space_keys:
            msg = "At least one valid Space ID is required."
            raise ValueError(msg)

        if metadata_filter:
            for space_key in space_keys:
                space_key["filter"] = metadata_filter

        request_body: dict[str, Any] = {
            "message": query,
            "spaceKeys": space_keys,
            "requestedSize": max_results,
            "fetchMemory": include_memory_definition,
        }

        post_config: dict[str, Any] = {}
        if reranker_id is not None:
            post_config["reranker_id"] = reranker_id
        if llm_id is not None:
            post_config["llm_id"] = llm_id
        if relevance_threshold is not None:
            post_config["relevance_threshold"] = relevance_threshold
        if llm_temperature is not None:
            post_config["llm_temp"] = llm_temperature
        if chronological_resort is not None:
            post_config["chronological_resort"] = chronological_resort
        if post_config:
            post_config.setdefault("max_results", max_results)
            request_body["postProcessor"] = {
                "name": self._CHAT_POSTPROCESSOR,
                "config": post_config,
            }

        start = time.monotonic()
        last_result: dict[str, Any] | None = None

        while True:
            with self._client() as client:
                response = client.post(
                    self._url("/v1/memories:retrieve"),
                    headers=self._ndjson_headers(),
                    json=request_body,
                )
                response.raise_for_status()

            results: list[dict[str, Any]] = []
            memories: list[dict[str, Any]] = []
            result_set_id = ""
            abstract_reply: dict[str, Any] | None = None

            response_text = response.text
            for line in response_text.strip().split("\n"):
                json_str = line.strip()
                if not json_str:
                    continue
                if json_str.startswith("data:"):
                    json_str = json_str[5:].strip()
                if json_str.startswith("event:") or not json_str:
                    continue

                try:
                    item = json.loads(json_str)

                    if item.get("resultSetBoundary"):
                        result_set_id = item["resultSetBoundary"].get("resultSetId", "")
                    elif item.get("memoryDefinition"):
                        memories.append(item["memoryDefinition"])
                    elif item.get("abstractReply"):
                        abstract_reply = item["abstractReply"]
                    elif item.get("retrievedItem"):
                        ri = item["retrievedItem"]
                        chunk_data = ri.get("chunk", {})
                        chunk = chunk_data.get("chunk", {})
                        results.append(
                            {
                                "chunkId": chunk.get("chunkId"),
                                "chunkText": chunk.get("chunkText"),
                                "memoryId": chunk.get("memoryId"),
                                "relevanceScore": chunk_data.get("relevanceScore"),
                                "memoryIndex": chunk_data.get("memoryIndex"),
                            }
                        )
                except (ValueError, KeyError):
                    continue

            last_result = {
                "success": True,
                "resultSetId": result_set_id,
                "results": results,
                "memories": memories,
                "totalResults": len(results),
                "query": query,
            }
            if abstract_reply is not None:
                last_result["abstractReply"] = abstract_reply

            if results or not wait_for_indexing:
                return last_result

            elapsed = time.monotonic() - start
            if elapsed >= max_wait_seconds:
                last_result["message"] = (
                    f"No results found after waiting {max_wait_seconds:g} "
                    "seconds for indexing. Memories may still be processing."
                )
                return last_result

            time.sleep(poll_interval)

    def get_memory(
        self,
        *,
        memory_id: str,
        include_content: bool = True,
    ) -> dict[str, Any]:
        """Fetch a specific memory by ID.

        Args:
            memory_id: The UUID of the memory.
            include_content: Whether to also fetch the original document content.

        Returns:
            A dictionary containing the memory metadata and optionally content.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        with self._client() as client:
            response = client.get(
                self._url(f"/v1/memories/{memory_id}"),
                headers=self._headers(),
            )
            response.raise_for_status()
            result: dict[str, Any] = {
                "success": True,
                "memory": response.json(),
            }

        if include_content:
            try:
                with self._client() as client:
                    content_response = client.get(
                        self._url(f"/v1/memories/{memory_id}/content"),
                        headers=self._headers(),
                    )
                    content_response.raise_for_status()
                    content_type = content_response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        result["content"] = content_response.json()
                    else:
                        result["content"] = content_response.text
            except (httpx.HTTPStatusError, ValueError) as e:
                result["contentError"] = f"Failed to fetch content: {e}"

        return result

    def list_memories(
        self,
        *,
        space_id: str,
        max_results: int | None = None,
        next_token: str | None = None,
        status_filter: str | None = None,
        include_content: bool = False,
        filter_expression: str | None = None,
        sort_by: str | None = None,
        sort_order: str | None = None,
    ) -> dict[str, Any]:
        """List memories in a space, with optional pagination and filtering.

        Args:
            space_id: The UUID of the space.
            max_results: Maximum results per page (clamped server-side).
            next_token: Opaque pagination token from a previous response.
            status_filter: Filter by processing status: PENDING, PROCESSING,
                COMPLETED, FAILED.
            include_content: Whether to include the original content in the
                response.
            filter_expression: Metadata filter expression (see GoodMem
                Filter Expressions reference).
            sort_by: Field to sort by. One of `created_at` or `updated_at`.
            sort_order: Sort direction. One of `ASCENDING` or `DESCENDING`.

        Returns:
            A dictionary with `memories` list and `nextToken`.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        params: dict[str, Any] = {}
        if max_results is not None:
            params["maxResults"] = max_results
        if next_token is not None:
            params["nextToken"] = next_token
        if status_filter is not None:
            params["statusFilter"] = status_filter
        if include_content:
            params["includeContent"] = True
        if filter_expression is not None:
            params["filter"] = filter_expression
        if sort_by is not None:
            params["sortBy"] = sort_by
        if sort_order is not None:
            params["sortOrder"] = sort_order

        with self._client() as client:
            response = client.get(
                self._url(f"/v1/spaces/{space_id}/memories"),
                headers=self._headers(),
                params=params,
            )
            response.raise_for_status()
            body = response.json()

        memories = body.get("memories", []) if isinstance(body, dict) else []
        next_token_resp = body.get("nextToken") if isinstance(body, dict) else None
        return {
            "success": True,
            "spaceId": space_id,
            "memories": memories,
            "totalMemories": len(memories),
            "nextToken": next_token_resp,
        }

    def delete_memory(self, *, memory_id: str) -> dict[str, Any]:
        """Delete a memory by ID.

        Args:
            memory_id: The UUID of the memory to delete.

        Returns:
            A dictionary confirming the deletion.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        with self._client() as client:
            response = client.delete(
                self._url(f"/v1/memories/{memory_id}"),
                headers=self._headers(),
            )
            response.raise_for_status()

        return {
            "success": True,
            "memoryId": memory_id,
            "message": "Memory deleted successfully",
        }

    def list_embedders(self) -> list[dict[str, Any]]:
        """List all available embedder models.

        Returns:
            A list of embedder dictionaries.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        with self._client() as client:
            response = client.get(
                self._url("/v1/embedders"),
                headers=self._headers(),
            )
            response.raise_for_status()
            body = response.json()

        if isinstance(body, list):
            return body
        return body.get("embedders", [])
