from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QTextBlockFormat, QTextCharFormat
from PySide6.QtWidgets import QDialog, QListWidgetItem, QMenu

from config import answer_detail_max_tokens, load_answer_detail
from core.chat_store import (
    delete_chat,
    list_chats,
    load_chat,
    new_chat,
    rename_chat,
    save_chat,
    title_from_first_message,
)
from core.llm_engine import is_server_running
from core.logger import log
from core.messages import (
    CHAT_COMPRESSION_FAILED,
    CHAT_CONTEXT_TOO_LARGE,
    CHAT_EMPTY_RESPONSE,
    CHAT_INTERRUPTED_ON_CLOSE,
    CHAT_LOCAL_MODEL_CONNECTION,
    CHAT_LOCAL_MODEL_MEMORY,
    CHAT_LOCAL_MODEL_STOPPED,
    CHAT_TOKEN_LIMIT_NO_RESPONSE,
    classify_by_substring,
)
from ui.chat.view import ChatItemWidget, RenameChatDialog, render_message_html
from ui.chat.workers import CompressionWorker, LLMWorker


_GENERATION_ERROR_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("out of memory", "oom", "memory", "metal", "kv-cache", "compute error"),
        CHAT_LOCAL_MODEL_MEMORY,
    ),
    (
        ("llm server is not running", "server is not running", "not running"),
        CHAT_LOCAL_MODEL_STOPPED,
    ),
    (
        ("connecterror", "readerror", "remoteprotocolerror", "connection", "disconnected"),
        CHAT_LOCAL_MODEL_CONNECTION,
    ),
    (
        ("context", "token", "length", "exceed", "too long", "413", "400"),
        CHAT_CONTEXT_TOO_LARGE,
    ),
)


def _friendly_generation_error(error: str) -> str:
    return classify_by_substring(error, _GENERATION_ERROR_PATTERNS) or error


class ChatController:
    def __init__(self, window):
        self.window = window

    def new_chat(self):
        log("UI", "Creating new chat")
        # Stop any in-flight generation and bump the generation token
        # so stale token-emit signals from the old chat are filtered
        # out by _is_current_generation — otherwise the previous chat's
        # response would stream into the new empty display.
        self._abort_active_generation()
        self.save_current_chat()
        self.window._current_chat = new_chat()
        self.window._history = []
        self.window._thinking_blocks.clear()
        self.window._chat_display.clear()
        self.refresh_chat_list()
        self.window._update_ctx_chip()

    def _abort_active_generation(self) -> None:
        """Cancel any in-flight LLMWorker and invalidate its callbacks.

        Used when switching chats: without this, the worker keeps
        streaming tokens after the user moved to a different chat, and
        those tokens get appended to the new chat's display because the
        chat display is shared.
        """
        worker = getattr(self.window, "_worker", None)
        if worker is not None and worker.isRunning():
            log("UI", "Aborting in-flight chat generation on chat switch")
            worker.stop()
        # Bump the token so any signals still in flight are dropped by
        # _is_current_generation when they reach the main thread.
        self.window._generation_token += 1
        self.window._stop_btn.setEnabled(False)
        self.window._send_btn.setEnabled(True)

    def save_current_chat(self):
        if self.window._history:
            self.window._current_chat["history"] = self.window._history
            save_chat(self.window._current_chat)

    def refresh_chat_list(self):
        self.window._chat_list.blockSignals(True)
        self.window._chat_list.clear()
        for chat in list_chats():
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, chat["id"])
            item.setData(Qt.ItemDataRole.DisplayRole, "")
            item.setData(Qt.ItemDataRole.UserRole + 1, chat["title"])
            item.setSizeHint(QSize(0, 42))
            self.window._chat_list.addItem(item)
            widget = ChatItemWidget(chat["id"], chat["title"])
            widget.rename_requested.connect(self.rename_chat_by_id)
            widget.delete_requested.connect(self.delete_chat_by_id)
            self.window._chat_list.setItemWidget(item, widget)
            if chat["id"] == self.window._current_chat["id"]:
                self.window._chat_list.setCurrentItem(item)
        self.window._chat_list.blockSignals(False)
        # Newly populated items must respect the active search filter.
        if hasattr(self.window, "_filter_chat_list"):
            self.window._filter_chat_list(self.window._chat_search.text())

    def rename_chat_by_id(self, chat_id: str, current_title: str):
        dlg = RenameChatDialog(current_title, self.window)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            title = dlg.title()
            if title:
                rename_chat(chat_id, title)
                if chat_id == self.window._current_chat["id"]:
                    self.window._current_chat["title"] = title
                self.refresh_chat_list()

    def delete_chat_by_id(self, chat_id: str):
        # If the user deletes the chat that's currently generating,
        # abort the worker first — otherwise its tokens stream into
        # the fresh empty chat we replace it with.
        if chat_id == self.window._current_chat["id"]:
            self._abort_active_generation()
        delete_chat(chat_id)
        if chat_id == self.window._current_chat["id"]:
            self.window._current_chat = new_chat()
            self.window._history = []
            self.window._thinking_blocks.clear()
            self.window._chat_display.clear()
        self.refresh_chat_list()

    def on_chat_selected(self, item):
        if item is None:
            return
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        if chat_id == self.window._current_chat["id"]:
            return
        log("UI", f"Switching to chat {chat_id}")
        # Same hazard as new_chat — stop any in-flight generation so its
        # tokens don't bleed into the chat we're switching to.
        self._abort_active_generation()
        self.save_current_chat()
        chat = load_chat(chat_id)
        if chat:
            self.window._current_chat = chat
            self.window._history = chat["history"]
            self.load_chat_into_display(chat)
            self.window._update_ctx_chip()

    def on_chat_rename(self, item):
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        widget = self.window._chat_list.itemWidget(item)
        current_title = widget.title if widget else ""
        self.rename_chat_by_id(chat_id, current_title)

    def on_chat_context_menu(self, pos):
        item = self.window._chat_list.itemAt(pos)
        if not item:
            return
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self.window)
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")
        action = menu.exec(self.window._chat_list.mapToGlobal(pos))
        if action == rename_action:
            self.on_chat_rename(item)
        elif action == delete_action:
            self.delete_chat_by_id(chat_id)

    def reset_format(self):
        cursor = self.window._chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.setBlockFormat(QTextBlockFormat())
        cursor.setCharFormat(QTextCharFormat())
        self.window._chat_display.setTextCursor(cursor)

    def load_chat_into_display(self, chat: dict):
        self.window._thinking_blocks.clear()
        self.window._chat_display.clear()
        first_msg = True
        for msg in chat["history"]:
            if msg["role"] == "user":
                self.reset_format()
                if not first_msg:
                    self.window._chat_display.append("<p>&nbsp;</p>")
                self.window._chat_display.append("<b style='color: #89b4fa;'>You</b>")
                self.window._chat_display.append(msg["content"])
                first_msg = False
            elif msg["role"] == "assistant":
                self.reset_format()
                if not first_msg:
                    self.window._chat_display.append("<p>&nbsp;</p>")
                model = msg.get("model", "")
                label = f"Assistant ({model})" if model else "Assistant"
                self.window._chat_display.append(f"<b style='color: #a6e3a1;'>{label}</b>")
                first_msg = False
                if thinking := msg.get("thinking"):
                    tid = self.window._thinking_id_counter
                    self.window._thinking_id_counter += 1
                    self.window._chat_display.append(
                        f"<a href='#thinking-{tid}' "
                        f"style='color: #6c7086; text-decoration: none;'>"
                        f"\u25b6 Thinking</a>"
                    )
                    cursor = self.window._chat_display.textCursor()
                    cursor.movePosition(cursor.MoveOperation.End)
                    self.window._thinking_blocks[tid] = {
                        "collapsed": True,
                        "header_block": cursor.blockNumber(),
                        "text": thinking,
                        "content_start": -1,
                        "content_end": -1,
                    }
                if msg.get("error"):
                    self.window._chat_display.append(
                        f"<i style='color: #f38ba8;'>{msg['content']}</i>"
                    )
                else:
                    html = render_message_html(msg["content"])
                    cursor = self.window._chat_display.textCursor()
                    cursor.movePosition(cursor.MoveOperation.End)
                    cursor.insertBlock()
                    cursor.setBlockFormat(QTextBlockFormat())
                    cursor.setCharFormat(QTextCharFormat())
                    cursor.insertHtml(html)
                    cursor.movePosition(cursor.MoveOperation.End)
                    self.window._chat_display.setTextCursor(cursor)

    def send_message(self):
        prompt = self.window._input_field.toPlainText().strip()
        if not prompt or not is_server_running():
            return
        if self.window.is_llm_busy():
            # llama-server runs --parallel 1; another worker (parse,
            # trends, health report) is already holding the slot, so a
            # new chat request would queue for minutes and look frozen.
            # Surface the same "wait or cancel" banner used by every
            # other section that hits this condition. Refuse before
            # consuming the prompt so the user's input stays where it
            # is and the chat history isn't polluted with an
            # unanswered turn.
            self.window._set_general_status(
                "Wait for the current operation to finish or cancel it, "
                "then try again."
            )
            return
        log("UI", f"_send_message: '{prompt[:80]}...'")
        self.window._input_field.clear()
        self.reset_format()
        if self.window._history:
            self.window._chat_display.append("<p>&nbsp;</p>")
        self.window._chat_display.append("<b style='color: #89b4fa;'>You</b>")
        self.window._chat_display.append(prompt)
        self.window._history.append({"role": "user", "content": prompt})
        if len(self.window._history) == 1:
            self.window._current_chat["title"] = title_from_first_message(prompt)
            # Persist before refresh: refresh_chat_list() reads titles from
            # disk via list_chats(), so without this save the sidebar would
            # keep showing "New Chat" until the assistant's reply finished
            # and triggered the next save.
            self.save_current_chat()
            self.refresh_chat_list()
        self.window._send_btn.setEnabled(False)
        self.window._compression_attempted = False
        self.launch_llm_worker(self.window._build_profile_context())

    def launch_llm_worker(self, context: str = ""):
        log(
            "UI",
            f"_launch_llm_worker: context={len(context)} chars, "
            f"history={len(self.window._history)} msgs",
        )
        self.window._thinking = True
        self.window._thinking_text = ""
        self.window._current_response = ""
        self.window._response_anchor = 0
        self.window._generation_stopped = False
        self.window._generation_token += 1
        token = self.window._generation_token
        answer_detail = load_answer_detail()
        max_tokens = answer_detail_max_tokens(answer_detail)
        use_rag = getattr(self.window, "_use_docs_btn", None)
        use_rag = True if use_rag is None else use_rag.isChecked()
        self.window._worker = LLMWorker(
            list(self.window._history),
            context=context,
            max_tokens=max_tokens,
            use_rag=use_rag,
            answer_detail=answer_detail,
        )
        self.window._worker.thinking_token.connect(
            lambda value, token=token: self.on_thinking_token(token, value)
        )
        self.window._worker.response_token.connect(
            lambda value, token=token: self.on_response_token(token, value)
        )
        self.window._worker.finished_generation.connect(
            lambda token=token: self.on_generation_done(token)
        )
        self.window._worker.cancelled_generation.connect(
            lambda token=token: self.on_generation_cancelled(token)
        )
        self.window._worker.error_occurred.connect(
            lambda error, token=token: self.on_generation_error(token, error)
        )
        self.window._stop_btn.setEnabled(True)
        self.window._worker.start()

    def stop_generation(self):
        log("UI", "Generation stopped by user")
        self.window._generation_stopped = True
        self.window._stop_btn.setEnabled(False)
        # LLMWorker handles the user's question. CompressionWorker may
        # have replaced it as the in-flight worker after a context
        # overflow (we re-prompt with summarised history); stopping
        # ONLY _worker would leave compression running for minutes.
        for worker in (
            getattr(self.window, "_worker", None),
            getattr(self.window, "_compression_worker", None),
        ):
            if worker is not None and worker.isRunning():
                worker.stop()

    def update_link_hover(self, pos):
        if self.window._hovered_link_range is not None:
            start, end = self.window._hovered_link_range
            cursor = self.window._chat_display.textCursor()
            cursor.setPosition(start)
            cursor.setPosition(end, cursor.MoveMode.KeepAnchor)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#6c7086"))
            cursor.mergeCharFormat(fmt)
            self.window._hovered_link_range = None

        if pos is None:
            return

        cursor = self.window._chat_display.cursorForPosition(pos)
        block = cursor.block()
        it = block.begin()
        while not it.atEnd():
            fragment = it.fragment()
            if fragment.isValid() and fragment.charFormat().anchorHref():
                start = fragment.position()
                end = start + fragment.length()
                cursor = self.window._chat_display.textCursor()
                cursor.setPosition(start)
                cursor.setPosition(end, cursor.MoveMode.KeepAnchor)
                fmt = QTextCharFormat()
                fmt.setForeground(QColor("#a6e3a1"))
                cursor.mergeCharFormat(fmt)
                self.window._hovered_link_range = (start, end)
                return
            it += 1

    def on_link_clicked(self, url):
        frag = url.fragment()
        if not frag.startswith("thinking-"):
            return
        try:
            tid = int(frag.split("-", 1)[1])
        except (ValueError, IndexError):
            return
        info = self.window._thinking_blocks.get(tid)
        if info:
            self.toggle_thinking(tid, info)

    def toggle_thinking(self, tid: int, info: dict):
        doc = self.window._chat_display.document()
        header_block = doc.findBlockByNumber(info["header_block"])
        if not header_block.isValid():
            return
        old_char_count = doc.characterCount()
        cursor = self.window._chat_display.textCursor()

        if info["collapsed"]:
            cursor.setPosition(header_block.position())
            cursor.movePosition(cursor.MoveOperation.EndOfBlock)
            cursor.insertBlock()
            start_block = cursor.blockNumber()
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#6c7086"))
            cursor.insertText(info["text"], fmt)
            end_block = cursor.blockNumber()
            info["content_start"] = start_block
            info["content_end"] = end_block
            info["collapsed"] = False
            inserted = end_block - start_block + 1
            for other_id, other in self.window._thinking_blocks.items():
                if other_id != tid and other["header_block"] > info["header_block"]:
                    other["header_block"] += inserted
                    if not other["collapsed"]:
                        other["content_start"] += inserted
                        other["content_end"] += inserted
        else:
            start_blk = doc.findBlockByNumber(info["content_start"])
            end_blk = doc.findBlockByNumber(info["content_end"])
            if start_blk.isValid() and end_blk.isValid():
                cursor.setPosition(start_blk.position() - 1)
                cursor.setPosition(
                    end_blk.position() + end_blk.length() - 1,
                    cursor.MoveMode.KeepAnchor,
                )
                cursor.removeSelectedText()
                removed = info["content_end"] - info["content_start"] + 1
                for other_id, other in self.window._thinking_blocks.items():
                    if other_id != tid and other["header_block"] > info["header_block"]:
                        other["header_block"] -= removed
                        if not other["collapsed"]:
                            other["content_start"] -= removed
                            other["content_end"] -= removed
            info["collapsed"] = True

        delta = doc.characterCount() - old_char_count
        if self.window._response_anchor > 0:
            self.window._response_anchor += delta

        header_block = doc.findBlockByNumber(info["header_block"])
        if header_block.isValid():
            cursor.setPosition(header_block.position())
            cursor.movePosition(cursor.MoveOperation.EndOfBlock, cursor.MoveMode.KeepAnchor)
            arrow = "\u25b6" if info["collapsed"] else "\u25bc"
            cursor.insertHtml(
                f"<a href='#thinking-{tid}' "
                f"style='color: #6c7086; text-decoration: none;'>"
                f"{arrow} Thinking...</a>"
            )

        self.window._chat_display.setTextCursor(cursor)
        self.window._chat_display.viewport().update()

    def _is_current_generation(self, token: int | None) -> bool:
        if token is None or token == self.window._generation_token:
            return True
        log("UI", f"Ignored stale generation signal: token={token}")
        return False

    def _is_current_compression(self, token: int | None) -> bool:
        if token is None or token == self.window._compression_token:
            return True
        log("UI", f"Ignored stale compression signal: token={token}")
        return False

    def on_thinking_token(self, token: int | None, value: str = ""):
        if isinstance(token, str) and not value:
            token, value = None, token
        if not self._is_current_generation(token):
            return
        first = not self.window._thinking_text
        self.window._thinking_text += value
        if first:
            tid = self.window._thinking_id_counter
            self.window._thinking_id_counter += 1
            self.window._current_thinking_id = tid
            self.window._chat_display.append("<p>&nbsp;</p>")
            label = (
                f"Assistant ({self.window._model_name})"
                if self.window._model_name
                else "Assistant"
            )
            self.window._chat_display.append(f"<b style='color: #a6e3a1;'>{label}</b>")
            self.window._chat_display.append(
                f"<a href='#thinking-{tid}' "
                f"style='color: #6c7086; text-decoration: none;'>"
                f"\u25b6 Thinking</a>"
            )
            cursor = self.window._chat_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.window._thinking_blocks[tid] = {
                "collapsed": True,
                "header_block": cursor.blockNumber(),
                "text": self.window._thinking_text,
                "content_start": -1,
                "content_end": -1,
            }
            return

        info = self.window._thinking_blocks[self.window._current_thinking_id]
        info["text"] = self.window._thinking_text
        if not info["collapsed"]:
            scrollbar = self.window._chat_display.verticalScrollBar()
            at_bottom = scrollbar.value() >= scrollbar.maximum() - 20
            cursor = self.window._chat_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#6c7086"))
            cursor.insertText(value, fmt)
            info["content_end"] = cursor.blockNumber()
            if at_bottom:
                scrollbar.setValue(scrollbar.maximum())

    def on_response_token(self, token: int | None, value: str = ""):
        if isinstance(token, str) and not value:
            token, value = None, token
        if not self._is_current_generation(token):
            return
        if self.window._thinking:
            self.window._thinking = False
            if self.window._thinking_text:
                info = self.window._thinking_blocks[self.window._current_thinking_id]
                info["text"] = self.window._thinking_text
            else:
                self.window._chat_display.append("<p>&nbsp;</p>")
                label = (
                    f"Assistant ({self.window._model_name})"
                    if self.window._model_name
                    else "Assistant"
                )
                self.window._chat_display.append(f"<b style='color: #a6e3a1;'>{label}</b>")
            cursor = self.window._chat_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.window._response_anchor = cursor.position()

        self.window._current_response += value
        html = render_message_html(self.window._current_response)
        scrollbar = self.window._chat_display.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 20
        saved_scroll = scrollbar.value()
        cursor = self.window._chat_display.textCursor()
        if self.window._response_anchor <= 0:
            log("UI", "WARNING: _response_anchor is 0, skipping response render")
            return
        cursor.setPosition(self.window._response_anchor)
        cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.setBlockFormat(QTextBlockFormat())
        cursor.insertBlock()
        cursor.setBlockFormat(QTextBlockFormat())
        cursor.insertHtml(html)
        self.window._chat_display.setTextCursor(cursor)
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(saved_scroll)

    def on_generation_done(self, token: int | None = None):
        if not self._is_current_generation(token):
            return
        log(
            "UI",
            f"Generation done, response={len(self.window._current_response)} chars, "
            f"stopped={self.window._generation_stopped}",
        )
        if self.window._thinking:
            self.window._thinking = False
            self.window._status_label.setText(f"Model: {self.window._model_name}")
        self.window._stop_btn.setEnabled(False)
        if self.window._generation_stopped:
            if self.window._history and self.window._history[-1]["role"] == "user":
                self.window._history.pop()
        elif self.window._current_response:
            msg = {
                "role": "assistant",
                "content": self.window._current_response,
                "model": self.window._model_name,
            }
            if self.window._thinking_text:
                msg["thinking"] = self.window._thinking_text
            self.window._history.append(msg)
            self.save_current_chat()
            self.refresh_chat_list()
        else:
            # Two empty-response paths: model spent its whole budget on
            # reasoning (Qwen 3 quirk — `thinking_text` populated but
            # `content` empty) vs returned literally nothing. Both leave
            # the user staring at a blank reply unless we surface a
            # message; previously only the thinking-only case was handled.
            error_text = (
                CHAT_TOKEN_LIMIT_NO_RESPONSE
                if self.window._thinking_text
                else CHAT_EMPTY_RESPONSE
            )
            self.window._chat_display.append(f"<i style='color: #f38ba8;'>{error_text}</i>")
            history_entry = {
                "role": "assistant",
                "content": error_text,
                "model": self.window._model_name,
                "error": True,
            }
            if self.window._thinking_text:
                history_entry["thinking"] = self.window._thinking_text
            self.window._history.append(history_entry)
            self.save_current_chat()
            self.refresh_chat_list()
        self.window._send_btn.setEnabled(True)
        self.window._update_ctx_chip()

    def on_generation_cancelled(self, token: int | None = None):
        if not self._is_current_generation(token):
            return
        log(
            "UI",
            f"Generation cancelled, response={len(self.window._current_response)} chars, "
            f"stopped={self.window._generation_stopped}",
        )
        self.window._thinking = False
        self.window._stop_btn.setEnabled(False)
        self.window._status_label.setText(f"Model: {self.window._model_name}")
        if self.window._generation_stopped:
            if self.window._history and self.window._history[-1]["role"] == "user":
                self.window._history.pop()
        else:
            error_text = CHAT_INTERRUPTED_ON_CLOSE
            self.window._chat_display.append(f"<i style='color: #f38ba8;'>{error_text}</i>")
            msg = {
                "role": "assistant",
                "content": error_text,
                "model": self.window._model_name,
                "error": True,
            }
            if self.window._thinking_text:
                msg["thinking"] = self.window._thinking_text
            self.window._history.append(msg)
            self.save_current_chat()
            self.refresh_chat_list()
        self.window._send_btn.setEnabled(True)
        self.window._update_ctx_chip()

    def on_generation_error(self, token: int | None, error: str = ""):
        if isinstance(token, str) and not error:
            token, error = None, token
        if not self._is_current_generation(token):
            return
        log("UI", f"Generation error: {error}")
        self.window._thinking = False
        self.window._stop_btn.setEnabled(False)
        self.window._status_label.setText(f"Model: {self.window._model_name}")
        error_lower = error.lower()
        is_ctx_overflow = any(
            kw in error_lower
            for kw in (
                "context",
                "token",
                "length",
                "exceed",
                "too long",
                "413",
                "400",
                "bad request",
            )
        )
        if (
            is_ctx_overflow
            and len(self.window._history) > 1
            and not self.window._compression_attempted
        ):
            self.window._pending_prompt = ""
            if self.window._history and self.window._history[-1]["role"] == "user":
                self.window._pending_prompt = self.window._history[-1]["content"]
                history_to_compress = self.window._history[:-1]
            else:
                history_to_compress = list(self.window._history)
            self.window._chat_display.append(
                "<b style='color: #a6e3a1;'>Assistant:</b> "
                "<i style='color: #6c7086;'>Compressing conversation history...</i>"
            )
            self.window._compression_attempted = True
            self.window._compression_token += 1
            compression_token = self.window._compression_token
            self.window._compression_worker = CompressionWorker(history_to_compress)
            self.window._compression_worker.finished.connect(
                lambda summary, token=compression_token: self.on_compression_done(
                    token, summary
                )
            )
            self.window._compression_worker.error_occurred.connect(
                lambda error, token=compression_token: self.on_compression_error(
                    token, error
                )
            )
            self.window._compression_worker.start()
        else:
            if self.window._history and self.window._history[-1]["role"] == "user":
                self.window._history.pop()
            display_error = _friendly_generation_error(error)
            self.window._chat_display.append(
                f"\n<i style='color: #f38ba8;'>Error: {display_error}</i>\n"
            )
            self.window._send_btn.setEnabled(True)

    def on_compression_done(self, token: int | None, summary: str = ""):
        if isinstance(token, str) and not summary:
            token, summary = None, token
        if not self._is_current_compression(token):
            return
        log("UI", f"Compression done, summary={len(summary)} chars, retrying with pending prompt")
        self.window._history = [
            {"role": "user", "content": f"[Summary of previous conversation]\n\n{summary}"},
            {
                "role": "assistant",
                "content": "Understood. I have the context from the previous conversation.",
            },
            {"role": "user", "content": self.window._pending_prompt},
        ]
        cursor = self.window._chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        self.window._chat_display.setTextCursor(cursor)
        self.launch_llm_worker()

    def on_compression_error(self, token: int | None, error: str = ""):
        if isinstance(token, str) and not error:
            token, error = None, token
        if not self._is_current_compression(token):
            return
        log("UI", f"Compression error: {error}")
        if self.window._history and self.window._history[-1]["role"] == "user":
            self.window._history.pop()
        cursor = self.window._chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        self.window._chat_display.setTextCursor(cursor)
        self.window._chat_display.append(
            "<b style='color: #a6e3a1;'>Assistant</b><br>"
            f"<i style='color: #f38ba8;'>{CHAT_COMPRESSION_FAILED}</i>"
        )
        self.window._send_btn.setEnabled(True)
