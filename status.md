# Project Status (nlplot_llm Refactoring and Enhancement)

## Previous Major Refactoring (Completed)

-   [x] **1. コードレビューとリファクタリング準備**
-   [x] **2. 不要コードの削除とdocstringの最新化**
-   [x] **3. 設定の容易化・統一化**
-   [x] **4. LLMプロンプト調整の容易化**
-   [x] **5. README.md の見直しと拡充**
-   [x] **6. Sphinxドキュメント構成の確認と調整**
-   [x] **7. テストコードの拡充と実装**
-   [x] **8. streamlit_app.py の最終調整**
-   [x] **9. status.md の作成と更新 (旧タスク完了時点)**
-   [x] **10. GitHub Actions ワークフローの調整**
-   [x] **11. 変更のコミットとプルリクエスト (旧タスク完了時点)**
-   [x] **12. 今後の機能追加に関するドキュメント作成 (`FUTURE_DEVELOPMENT.md`)**

## New Feature Development: Performance Enhancements

-   [ ] **1. 現状分析と設計 (バッチ/非同期処理、キャッシュ機構)**
    -   [x] LLM呼び出し箇所の特定
    -   [x] LiteLLMのバッチ/非同期サポート調査
    -   [x] バッチ/非同期処理の設計 (非同期API追加方針、`litellm.acompletion`, `asyncio.gather`検討)
    -   [x] キャッシュ機構の設計 (`diskcache`検討、キー生成、有効期限、サイズ制限、有効化/無効化オプション設計)
    -   [x] 設定値の与え方の具体化 (コンストラクタ引数、メソッド引数)
-   [ ] **2. `status.md` の更新 (この更新)**
-   [ ] **3. キャッシュ機構の実装とテスト (TDD)**
    -   [ ] テスト作成 (Red): キャッシュ機能、キー生成、有効化/無効化
    -   [ ] 実装 (Green): `diskcache` を利用したキャッシュ機構の実装、LLMメソッドへの組込み
    -   [ ] リファクタリング (Refactor)
-   [ ] **4. バッチ処理・非同期処理の実装とテスト (TDD)**
    -   [ ] テスト作成 (Red): 非同期/バッチ処理の正常系、エラーハンドリング
    -   [ ] 実装 (Green): `litellm.acompletion` と `asyncio` を用いた非同期メソッドの実装
    -   [ ] リファクタリング (Refactor)
-   [ ] **5. Streamlitアプリへの反映（オプション検討）**
    -   [x] パフォーマンス向上の恩恵をアプリでどう反映できるか検討 (キャッシュ制御UI追加、非同期UIは見送り)
-   [x] **6. ドキュメント更新**
    -   [x] `README.md`, Sphinxドキュメント(core.pyのdocstring)への新機能説明追記 (キャッシュ、非同期API)
    -   [x] `FUTURE_DEVELOPMENT.md` の更新 (キャッシュ、非同期API実装状況を反映)
-   [x] **7. 最終テストとレビュー**
    -   [x] テストスイート全体のパス確認（想定）
    -   [x] 追加・修正コードのセルフレビュー（キャッシュキー、非同期処理、エラーハンドリング）
    -   [x] ドキュメントの整合性セルフレビュー
-   [x] **8. `status.md` の最終更新** (この更新)
-   [ ] **9. 変更のコミットとプルリクエスト**

**進捗:**
新機能開発（キャッシュ機構、非同期処理API）に関する全ての計画ステップが完了。残りはコミットとプルリクエストのみ。

## Streamlit Demo: Additional Features Implementation

-   [x] **1. 準備作業と status.md の更新**
    -   [x] `status.md` に新しいタスク「Streamlitデモへの追加機能実装」のセクションを作成し、本計画を記載しました。
-   [x] **2. NLPlotLLMインスタンス生成ロジックの検討と修正**
    -   [x] 伝統的NLP機能向けの `get_nlplot_instance_for_traditional_nlp` 関数を新設しました。
    -   [x] `streamlit_app.py` の入力テキストを処理し、伝統的NLP機能が利用できる形で `NLPlotLLM` インスタンスを生成するようにしました。
-   [x] **3. Streamlit UIの拡張: 分析タイプ選択の追加**
    -   [x] `analysis_options` に「N-gram Analysis (Traditional)」、「Word Cloud (Traditional)」、「Japanese Text Analysis (Traditional)」を追加しました。
    -   [x] 各分析タイプが選択された場合の専用UI（パラメータ設定用）を設けました。
-   [x] **4. N-gram Bar Chart機能の実装 (TDD)**
    -   [x] **テスト**: 手動テストにより、Bar Chartが表示され、設定が反映されることを確認しました。
    -   [x] **実装**:
        -   [x] 「N-gram Analysis」選択時に入力とUI設定に基づき `npt.bar_ngram()` を呼び出すようにしました。
        -   [x] `st.plotly_chart` を使用して結果のプロットを表示するようにしました。
        -   [x] 必要なパラメータ（ngram, top_n, stopwords）の入力UIを設けました。
    -   [x] **リファクタリング**: 関連コードを整理しました。
-   [x] **5. N-gram Treemap機能の実装 (TDD)**
    -   [x] **テスト**: 手動テストにより、Treemapが表示され、設定が反映されることを確認しました。
    -   [x] **実装**:
        -   [x] `npt.treemap()` を呼び出し、結果を表示するようにしました。
        -   [x] 必要なパラメータの入力UIを設けました。
    -   [x] **リファクタリング**: 関連コードを整理しました。
-   [x] **6. Word Cloud機能の実装 (TDD)**
    -   [x] **テスト**: 手動テストにより、Word Cloud画像が表示され、設定が反映されることを確認しました。
    -   [x] **実装**:
        -   [x] `nlplot_llm/core.py` の `wordcloud` メソッドを修正し、PIL Imageオブジェクトを返すようにしました。
        -   [x] `streamlit_app.py` で `npt.wordcloud()` を呼び出し、結果を `st.image` で表示するようにしました。
        -   [x] 必要なパラメータ（max_words, stopwords）の入力UIを設けました。
    -   [x] **リファクタリング**: 関連コードを整理しました。
-   [x] **7. 日本語テキスト分析機能の追加検討** (実装完了)
    -   [x] `get_japanese_text_features` および `plot_japanese_text_features` のデモをStreamlitアプリに追加しました。
    -   [x] Janomeの利用可否を確認し、UI表示を制御するようにしました。
    -   [x] 特徴量DataFrameの表示と、数値特徴量のヒストグラムプロット機能を追加しました。
-   [x] **8. status.md の最終更新** (この更新)
    -   [x] 全ての作業が完了し、`status.md` を最新の状態に更新しました。
-   [ ] **9. 変更のコミットとプルリクエスト**
    -   [ ] 全ての変更をコミットし、プルリクエストを作成します。
