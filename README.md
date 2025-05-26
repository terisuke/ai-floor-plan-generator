# AI フロアプラン生成アプリ

このアプリケーションは、AI を使用してフロアプラン（間取り図）を自動生成するシステムです。

## ローカルでの実行方法

**前提条件:** 
- Node.js (v18以上推奨)
- Python 3.8以上
- Mac の場合: M1/M2 チップ対応（MPS アクセラレーション使用）

### セットアップ手順

1. **依存関係のインストール:**
   ```bash
   # フロントエンドの依存関係
   npm install
   
   # バックエンドの依存関係
   cd backend
   npm install
   cd ..
   
   # Python の依存関係
   cd inference
   pip3 install -r requirements.txt
   cd ..
   ```

2. **アプリケーションの起動:**
   ```bash
   # ターミナル 1: フロントエンドサーバー
   npm run dev
   
   # ターミナル 2: バックエンドサーバー
   cd backend
   npm run dev
   ```

3. **ブラウザでアクセス:**
   http://localhost:5173

## 機能

- グリッドサイズ（幅×高さ）を指定してフロアプランを生成
- AI モデル（Stable Diffusion）を使用した高品質な間取り図生成
- LoRA でファインチューニングされたモデルによる建築スタイルの最適化

## アーキテクチャ

- **フロントエンド:** React + TypeScript + Vite
- **バックエンド:** Node.js + Express + TypeScript
- **AI 推論:** Python + Stable Diffusion + LoRA

## トレーニング

カスタムデータセットでモデルをファインチューニングする場合は、[training/README.md](training/README.md) を参照してください。
