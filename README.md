# UchidaLab Marp Themes
内田研のゼミ・輪講・プロジェクト研究での発表資料作成に利用できるMarpテーマです。

HTMLを書くことで利用できる独自の機能をいくつか追加しています。

## 導入方法
1. Visual Studio Codeの拡張機能[Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)をインストールします。

2. `themes`内の利用したいテーマのcssファイルをダウンロードします（scssファイルをダウンロードする必要はありません）。

3. ダウンロードしたcssファイルを作業ディレクトリに配置します。

4. Marp for VS Codeの設定を次のように変更します。
    - `Markdown > Marp: Enable HTML`にチェックをつけます。
    - `Markdown > Marp: Themes`にcssファイルへのパス（theme_Eの場合は`./theme_E.css`）を追加します。

5. .mdファイルの冒頭で`theme: theme_E`のようにテーマを指定することで、ダウンロードしたテーマが適用されます。

## 各テーマの使い方
全てのテーマに共通する基本的な使い方は`samples/sample.md`を参照してください。

テーマごとに独自の機能については、`samples`にある実際の利用例を参考にしてください。　
