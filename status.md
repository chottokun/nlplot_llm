# Project Status (nlplot_llm Refactoring and Enhancement)

## ToDo List

-   [x] **1. コードレビューとリファクタリング準備**
    -   [x] `nlplot_llm/core.py` を詳細にレビューし、不要なコメント、古いLangChain関連のコード（TextSplittersを除く）、デッドコードを特定。
    -   [x] docstring を確認し、現在のLiteLLMベースの実装と一致しているか、最新化が必要な箇所を特定。
    -   [x] `tests/test_nlplot_llm_client_old_langchain.py` が完全に不要であることを確認。
-   [x] **2. 不要コードの削除とdocstringの最新化**
    -   [x] 特定した不要なコードコメントやデッドコードを `nlplot_llm/core.py` から削除。
    -   [x] `nlplot_llm/core.py` 内のdocstringを最新化し、LiteLLMベースの引数や動作を正確に反映。
    -   [x] `tests/test_nlplot_llm_client_old_langchain.py` を削除。
-   [x] **3. 設定の容易化・統一化**
    -   [x] `nlplot_llm/core.py` と `streamlit_app.py` を確認し、LLM関連の設定がユーザーにとって分かりやすく、一貫性のある方法で提供されているかを確認。
    -   [x] Streamlitアプリケーション (`streamlit_app.py`) でのLLM設定インターフェースが直感的であるか、改善。 (`Max Tokens`追加、ガイダンス改善)
    -   [x] 設定方法に関するドキュメント（READMEやdocstring）を改善。
-   [x] **4. LLMプロンプト調整の容易化**
    -   [x] `nlplot_llm/core.py` のLLM関連メソッドで、プロンプトテンプレートが外部から容易に変更可能になっているか確認。 (対応済み)
    -   [x] `streamlit_app.py` で、ユーザーがプロンプトを試行錯誤しやすいUI/UXになっているか確認し、改善。(感情分析、テキスト分類にもプロンプト編集エリア追加)
-   [x] **5. README.md の見直しと拡充**
    -   [x] 現在の `README.md` を精査し、LiteLLMへの移行、設定方法の変更、プロンプト調整の容易化といった最新の状況を反映。
    -   [x] インストール方法、必須・推奨ライブラリ、各機能の使い方、設定例、実行例などを網羅的に記述し、ユーザーが容易に利用開始できるように改善。
    -   [x] Streamlitデモアプリケーションの実行方法や設定についても詳細に記載。
    -   [x] Sphinxドキュメントの生成方法について追記。
    -   [x] API利用例のモデル名を `openai/gpt-4o`, `ollama/gemma` に更新。
-   [x] **6. Sphinxドキュメント構成の確認と調整**
    -   [x] `docs/` ディレクトリ内のSphinx設定 (`conf.py`, `index.rst`など) を確認し、現在のプロジェクト構成やモジュール構成と一致するように調整。
    -   [x] `nlplot_llm` モジュールが正しくAPIドキュメントとして生成されるように設定を調整。
    -   [x] `setup.py` の `package_data` を修正。
-   [x] **7. テストコードの拡充と実装**
    -   [x] **LiteLLM連携テスト:** `test_nlplot_llm_categorize.py`, `test_nlplot_llm_sentiment.py`, `test_nlplot_llm_summarize.py` をレビューし、LiteLLMの主要な機能（異なるプロバイダーやモデルの呼び出し、エラーハンドリングなど）が網羅されるようにテストケースを追加。
    -   [x] **エッジケーステスト:** 各機能について、エッジケース（空の入力、不正な形式の入力など）を考慮したテストを追加。
    -   [x] **テキストチャンキングテスト:** `test_nlplot_llm_utils.py` の `_chunk_text` のテストを拡充し、異なるチャンキング戦略やパラメータでの動作、依存ライブラリがない場合のフォールバック動作を確認。
    -   [x] 既存のテスト (`test_nlplot.py`) も見直し、カバレッジを確認。 (大きな追加は不要と判断)
-   [x] **8. streamlit_app.py の最終調整**
    -   [x] ステップ3、4で行った検討結果に基づき、`streamlit_app.py` のUI/UXを改善。
    -   [x] 設定オプションの明確化、プロンプト調整機能のUI調整。
    -   [x] README.md との整合性を最終確認。
-   [x] **9. status.md の作成と更新**
    -   [x] プロジェクトルートに `status.md` ファイルを作成。
    -   [x] ToDoリストを逐次更新しながら進める。 (この更新を含む)
-   [x] **10. GitHub Actions ワークフローの調整**
    -   [x] `pythonpackage.yml` のPythonテストバージョンを更新（例: 3.7, 3.8, 3.9, 3.10, 3.11）。Python 3.6のサポートは終了。
    -   [x] 上記に合わせて `setup.py` の `python_requires` を更新（例: `>=3.7`）。
    -   [x] `pythonpackage.yml` の依存関係インストールステップで、`requirements.txt` もインストールされるように修正。
    -   [x] `python-publish.yml` のPythonバージョンを、テストで使用している主要なバージョン（例: `3.8`）に固定。
-   [x] **11. 変更のコミットとプルリクエスト**
    -   [x] すべての変更が完了し、テストがパスすることを確認後、変更をコミット。
-   [x] **12. 今後の機能追加に関するドキュメント作成**
    -   [x] `FUTURE_DEVELOPMENT.md` を作成し、今後の展望や開発方針を記載。

**進捗:**
全ての計画ステップが完了しました。
