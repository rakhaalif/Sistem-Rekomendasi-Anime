"""
Sistem Rekomendasi Anime - Proyek Akhir Machine Learning Terapan
Menggunakan Content-Based Filtering dan Collaborative Filtering
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class AnimeRecommendationSystem:
    def __init__(self, anime_path, rating_path):
        """
        Inisialisasi sistem rekomendasi anime
        
        Parameters:
        anime_path (str): Path ke file anime.csv
        rating_path (str): Path ke file rating.csv
        """
        try:
            self.anime_df = pd.read_csv(anime_path)
            self.rating_df = pd.read_csv(rating_path)
            self.content_similarity_matrix = None
            self.collaborative_model = None
            self.user_factors = None
            self.item_factors = None
            self.merged_df = None
            print(f"✅ Data berhasil dimuat: {len(self.anime_df)} anime, {len(self.rating_df)} ratings")
        except FileNotFoundError as e:
            print(f"❌ File tidak ditemukan: {e}")
            raise
        except Exception as e:
            print(f"❌ Error saat memuat data: {e}")
            raise
        
    def data_understanding(self):
        """Analisis dan pemahaman data"""
        print("=== DATA UNDERSTANDING ===")
        print(f"Anime dataset shape: {self.anime_df.shape}")
        print(f"Rating dataset shape: {self.rating_df.shape}")
        
        return self.anime_df.head(), self.rating_df.head()
    
    def data_preparation(self):
        """Preprocessing dan cleaning data"""
        print("🔄 Memproses dan membersihkan data...")
        
        # Handle missing values
        self.anime_df['genre'] = self.anime_df['genre'].fillna('Unknown')
        self.anime_df['type'] = self.anime_df['type'].fillna('Unknown')
        self.anime_df['rating'] = self.anime_df['rating'].fillna(0)
        self.anime_df['episodes'] = self.anime_df['episodes'].fillna(0)
        
        # Remove ratings with -1 (user didn't watch)
        self.rating_df = self.rating_df[self.rating_df['rating'] != -1]
        
        # Merge datasets
        self.merged_df = self.rating_df.merge(self.anime_df, on='anime_id', how='left')
        
        print("✅ Data preparation selesai!")
        return self.merged_df.head()
    
    def content_based_filtering(self):
        """Implementasi Content-Based Filtering"""
        print("🎯 Melatih model Content-Based Filtering...")
        
        # Create content features
        self.anime_df['content_features'] = (
            self.anime_df['genre'].astype(str) + ' ' + 
            self.anime_df['type'].astype(str)
        )
        
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(stop_words='english', lowercase=True, max_features=5000)
        tfidf_matrix = tfidf.fit_transform(self.anime_df['content_features'])
        
        # Calculate cosine similarity
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
        
        print("✅ Content-based model berhasil dilatih!")
        
    def collaborative_filtering(self):
        """Implementasi Collaborative Filtering menggunakan SVD"""
        print("👥 Melatih model Collaborative Filtering...")
        
        # Untuk efisiensi, gunakan sample data
        user_counts = self.rating_df['user_id'].value_counts()
        anime_counts = self.rating_df['anime_id'].value_counts()
        
        # Filter user dan anime dengan minimum interaksi
        min_user_ratings = 20
        min_anime_ratings = 50
        
        active_users = user_counts[user_counts >= min_user_ratings].index
        popular_anime = anime_counts[anime_counts >= min_anime_ratings].index
        
        # Filter rating data
        filtered_ratings = self.rating_df[
            (self.rating_df['user_id'].isin(active_users)) & 
            (self.rating_df['anime_id'].isin(popular_anime))
        ]
        
        if len(filtered_ratings) == 0:
            print("⚠️ Tidak cukup data untuk collaborative filtering")
            return
        
        # Create user-item matrix
        user_item_matrix = filtered_ratings.pivot_table(
            index='user_id', 
            columns='anime_id', 
            values='rating'
        ).fillna(0)
        
        # SVD for dimensionality reduction
        n_components = min(50, min(user_item_matrix.shape) - 1)
        self.collaborative_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.collaborative_model.fit_transform(user_item_matrix)
        self.item_factors = self.collaborative_model.components_.T
        
        # Store the matrix for later use
        self.user_item_matrix = user_item_matrix
        
        print("✅ Collaborative filtering model berhasil dilatih!")
        
    def get_content_recommendations(self, anime_title, n_recommendations=10):
        """Mendapatkan rekomendasi berdasarkan content"""
        try:
            # Find anime index
            anime_match = self.anime_df[self.anime_df['name'] == anime_title]
            if anime_match.empty:
                return f"Anime '{anime_title}' tidak ditemukan dalam database"
            
            anime_idx = anime_match.index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.content_similarity_matrix[anime_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top recommendations
            recommended_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
            
            recommendations = self.anime_df.iloc[recommended_indices].copy()
            recommendations['similarity_score'] = [i[1] for i in sim_scores[1:n_recommendations+1]]
            
            return recommendations[['name', 'genre', 'type', 'rating', 'similarity_score']]
            
        except Exception as e:
            return f"Error: {str(e)}"

    def get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """Mendapatkan rekomendasi berdasarkan collaborative filtering"""
        try:
            if not hasattr(self, 'user_item_matrix'):
                return "Model collaborative filtering belum dilatih"
            
            if user_id not in self.user_item_matrix.index:
                return f"User {user_id} tidak ditemukan dalam data training"
            
            # Get user index
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            
            # Predict ratings for all anime
            user_predictions = np.dot(self.user_factors[user_idx], self.collaborative_model.components_)
            
            # Get anime that user hasn't rated
            user_rated_anime = self.user_item_matrix.loc[user_id]
            unrated_anime_indices = np.where(user_rated_anime == 0)[0]
            
            # Get predictions for unrated anime
            anime_predictions = []
            for anime_idx in unrated_anime_indices:
                anime_id = self.user_item_matrix.columns[anime_idx]
                predicted_rating = user_predictions[anime_idx]
                anime_predictions.append((anime_id, predicted_rating))
            
            # Sort by predicted rating
            anime_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Get top recommendations
            top_anime_ids = [anime_id for anime_id, _ in anime_predictions[:n_recommendations]]
            predicted_ratings = [rating for _, rating in anime_predictions[:n_recommendations]]
            
            # Get anime details
            recommended_anime = self.anime_df[self.anime_df['anime_id'].isin(top_anime_ids)].copy()
            
            if recommended_anime.empty:
                return "Tidak ada rekomendasi yang ditemukan"
            
            # Add predicted ratings
            rating_dict = dict(anime_predictions[:n_recommendations])
            recommended_anime['predicted_rating'] = recommended_anime['anime_id'].map(rating_dict)
            
            return recommended_anime[['name', 'genre', 'type', 'rating', 'predicted_rating']].sort_values('predicted_rating', ascending=False)
            
        except Exception as e:
            return f"Error: {str(e)}"

    def evaluate_model(self):
        """Evaluasi model menggunakan RMSE"""
        print("📊 Mengevaluasi model...")
        
        try:
            # Sample data for evaluation
            sample_data = self.rating_df.sample(n=min(10000, len(self.rating_df)), random_state=42)
            
            # Split data
            train_data, test_data = train_test_split(sample_data, test_size=0.2, random_state=42)
            
            # Create user-item matrix for train data
            train_matrix = train_data.pivot_table(
                index='user_id', 
                columns='anime_id', 
                values='rating'
            ).fillna(0)
            
            # Fit SVD model
            n_components = min(50, min(train_matrix.shape) - 1)
            svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            user_factors = svd_model.fit_transform(train_matrix)
            
            # Predict ratings
            predicted_matrix = np.dot(user_factors, svd_model.components_)
            
            # Calculate RMSE
            actual_ratings = []
            predicted_ratings = []
            
            for _, row in test_data.iterrows():
                user_id = row['user_id']
                anime_id = row['anime_id']
                actual_rating = row['rating']
                
                if user_id in train_matrix.index and anime_id in train_matrix.columns:
                    user_idx = train_matrix.index.get_loc(user_id)
                    anime_idx = train_matrix.columns.get_loc(anime_id)
                    predicted_rating = predicted_matrix[user_idx, anime_idx]
                    
                    actual_ratings.append(actual_rating)
                    predicted_ratings.append(predicted_rating)
            
            if len(actual_ratings) == 0:
                return 0.0
            
            rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
            return rmse
            
        except Exception as e:
            print(f"Error dalam evaluasi: {str(e)}")
            return 0.0

# Fungsi-fungsi interaktif
def interactive_recommendation_system():
    """Sistem rekomendasi interaktif dengan input user"""
    print("\n" + "="*80)
    print("🎌 SISTEM REKOMENDASI ANIME INTERAKTIF 🎌")
    print("="*80)
    
    # Initialize system
    print("Memuat sistem rekomendasi...")
    try:
        recommender = AnimeRecommendationSystem('dataset/anime.csv', 'dataset/rating.csv')
        recommender.data_preparation()
        recommender.content_based_filtering()
        recommender.collaborative_filtering()
        print("✅ Sistem siap digunakan!\n")
    except Exception as e:
        print(f"❌ Error saat inisialisasi: {str(e)}")
        return
    
    while True:
        print("\n" + "="*60)
        print("MENU SISTEM REKOMENDASI")
        print("="*60)
        print("1. 🎯 Rekomendasi berdasarkan Anime (Content-Based)")
        print("2. 👤 Rekomendasi berdasarkan User (Collaborative)")
        print("3. 📊 Lihat daftar Anime populer")
        print("4. 👥 Lihat daftar User aktif")
        print("5. 🔍 Cari Anime")
        print("6. 📈 Evaluasi Model")
        print("7. ❌ Keluar")
        print("-"*60)
        
        try:
            choice = input("Pilih menu (1-7): ").strip()
            
            if choice == '1':
                content_based_menu(recommender)
            elif choice == '2':
                collaborative_menu(recommender)
            elif choice == '3':
                show_popular_anime(recommender)
            elif choice == '4':
                show_active_users(recommender)
            elif choice == '5':
                search_anime(recommender)
            elif choice == '6':
                evaluate_model_menu(recommender)
            elif choice == '7':
                print("\n🙏 Terima kasih telah menggunakan sistem rekomendasi anime!")
                break
            else:
                print("❌ Pilihan tidak valid! Pilih 1-7.")
                
        except KeyboardInterrupt:
            print("\n\n🙏 Terima kasih telah menggunakan sistem rekomendasi anime!")
            break
        except Exception as e:
            print(f"❌ Terjadi error: {str(e)}")

def content_based_menu(recommender):
    """Menu untuk content-based filtering"""
    print("\n" + "="*60)
    print("🎯 REKOMENDASI BERDASARKAN ANIME (CONTENT-BASED)")
    print("="*60)
    
    while True:
        print("\nPilihan:")
        print("1. Masukkan nama anime secara manual")
        print("2. Pilih dari daftar anime populer")
        print("3. Pilih anime random")
        print("4. Kembali ke menu utama")
        
        sub_choice = input("\nPilih opsi (1-4): ").strip()
        
        if sub_choice == '1':
            anime_name = input("\nMasukkan nama anime: ").strip()
            if anime_name:
                get_content_recommendations_interactive(recommender, anime_name)
            else:
                print("❌ Nama anime tidak boleh kosong!")
                
        elif sub_choice == '2':
            select_from_popular_anime(recommender)
            
        elif sub_choice == '3':
            random_anime = recommender.anime_df.sample(1).iloc[0]['name']
            print(f"\n🎲 Anime random terpilih: {random_anime}")
            get_content_recommendations_interactive(recommender, random_anime)
            
        elif sub_choice == '4':
            break
        else:
            print("❌ Pilihan tidak valid!")

def collaborative_menu(recommender):
    """Menu untuk collaborative filtering"""
    print("\n" + "="*60)
    print("👤 REKOMENDASI BERDASARKAN USER (COLLABORATIVE)")
    print("="*60)
    
    while True:
        print("\nPilihan:")
        print("1. Masukkan User ID secara manual")
        print("2. Pilih dari daftar user aktif")
        print("3. Pilih user random")
        print("4. Kembali ke menu utama")
        
        sub_choice = input("\nPilih opsi (1-4): ").strip()
        
        if sub_choice == '1':
            try:
                user_id = int(input("\nMasukkan User ID: ").strip())
                get_collaborative_recommendations_interactive(recommender, user_id)
            except ValueError:
                print("❌ User ID harus berupa angka!")
                
        elif sub_choice == '2':
            select_from_active_users(recommender)
            
        elif sub_choice == '3':
            random_user = recommender.rating_df.sample(1).iloc[0]['user_id']
            print(f"\n🎲 User random terpilih: {random_user}")
            get_collaborative_recommendations_interactive(recommender, random_user)
            
        elif sub_choice == '4':
            break
        else:
            print("❌ Pilihan tidak valid!")

def get_content_recommendations_interactive(recommender, anime_name):
    """Mendapatkan dan menampilkan rekomendasi content-based secara interaktif"""
    print(f"\n🔍 Mencari rekomendasi untuk: '{anime_name}'...")
    
    # Cek apakah anime ada dalam database
    matching_anime = recommender.anime_df[
        recommender.anime_df['name'].str.contains(anime_name, case=False, na=False)
    ]
    
    if matching_anime.empty:
        print(f"❌ Anime '{anime_name}' tidak ditemukan!")
        print("\n💡 Saran: Coba gunakan menu pencarian atau pilih dari daftar anime populer")
        return
    
    # Jika ada beberapa match, biarkan user memilih
    if len(matching_anime) > 1:
        print(f"\n🔍 Ditemukan {len(matching_anime)} anime yang cocok:")
        for i, (_, anime) in enumerate(matching_anime.head(10).iterrows(), 1):
            print(f"{i}. {anime['name']} ({anime['type']}, {anime['genre']})")
        
        try:
            choice = int(input(f"\nPilih anime (1-{min(10, len(matching_anime))}): ")) - 1
            if 0 <= choice < len(matching_anime):
                selected_anime = matching_anime.iloc[choice]['name']
            else:
                print("❌ Pilihan tidak valid!")
                return
        except ValueError:
            print("❌ Input harus berupa angka!")
            return
    else:
        selected_anime = matching_anime.iloc[0]['name']
    
    # Dapatkan rekomendasi
    recommendations = recommender.get_content_recommendations(selected_anime, 10)
    
    if isinstance(recommendations, str):
        print(f"❌ {recommendations}")
        return
    
    # Tampilkan informasi anime asli
    original_anime = recommender.anime_df[recommender.anime_df['name'] == selected_anime].iloc[0]
    print(f"\n📺 ANIME YANG DIPILIH:")
    print(f"   Nama: {original_anime['name']}")
    print(f"   Genre: {original_anime['genre']}")
    print(f"   Type: {original_anime['type']}")
    print(f"   Rating: {original_anime['rating']}")
    print(f"   Episodes: {original_anime['episodes']}")
    
    # Tampilkan rekomendasi
    print(f"\n🎯 REKOMENDASI ANIME SERUPA:")
    print("-" * 80)
    
    for i, (_, anime) in enumerate(recommendations.iterrows(), 1):
        print(f"{i:2d}. {anime['name']}")
        print(f"     Genre: {anime['genre']}")
        print(f"     Type: {anime['type']} | Rating: {anime['rating']}")
        print(f"     Similarity Score: {anime['similarity_score']:.4f}")
        print()
    
    # Analisis genre
    print("📊 ANALISIS GENRE DALAM REKOMENDASI:")
    all_genres = []
    for genre_list in recommendations['genre']:
        if pd.notna(genre_list):
            all_genres.extend(str(genre_list).split(', '))
    
    genre_counts = Counter(all_genres)
    for genre, count in genre_counts.most_common(5):
        print(f"   • {genre}: {count} anime")

def get_collaborative_recommendations_interactive(recommender, user_id):
    """Mendapatkan dan menampilkan rekomendasi collaborative secara interaktif"""
    print(f"\n🔍 Mencari rekomendasi untuk User {user_id}...")
    
    # Cek apakah user ada dalam database
    if user_id not in recommender.rating_df['user_id'].values:
        print(f"❌ User {user_id} tidak ditemukan dalam database!")
        print("💡 Saran: Coba pilih dari daftar user aktif")
        return
    
    # Tampilkan profil user
    user_data = recommender.merged_df[recommender.merged_df['user_id'] == user_id]
    print(f"\n👤 PROFIL USER {user_id}:")
    print(f"   Total anime yang dirating: {len(user_data)}")
    print(f"   Rating rata-rata: {user_data['rating_x'].mean():.2f}")
    
    # Tampilkan anime favorit user
    top_rated = user_data.nlargest(5, 'rating_x')
    print(f"\n⭐ TOP 5 ANIME FAVORIT USER:")
    for i, (_, anime) in enumerate(top_rated.iterrows(), 1):
        print(f"{i}. {anime['name']} (Rating: {anime['rating_x']})")
    
    # Genre favorit
    all_genres = []
    for genre_list in user_data['genre']:
        if pd.notna(genre_list):
            all_genres.extend(str(genre_list).split(', '))
    
    genre_counts = Counter(all_genres)
    print(f"\n🎭 GENRE FAVORIT:")
    for genre, count in genre_counts.most_common(3):
        print(f"   • {genre}: {count} anime")
    
    # Dapatkan rekomendasi
    recommendations = recommender.get_collaborative_recommendations(user_id, 10)
    
    if isinstance(recommendations, str):
        print(f"❌ {recommendations}")
        return
    
    # Tampilkan rekomendasi
    print(f"\n🎯 REKOMENDASI UNTUK USER {user_id}:")
    print("-" * 80)
    
    for i, (_, anime) in enumerate(recommendations.iterrows(), 1):
        pred_rating = anime.get('predicted_rating', 'N/A')
        print(f"{i:2d}. {anime['name']}")
        print(f"     Genre: {anime['genre']}")
        print(f"     Type: {anime['type']} | Rating: {anime['rating']}")
        if pred_rating != 'N/A':
            print(f"     Predicted Rating: {pred_rating:.2f}")
        print()

def select_from_popular_anime(recommender):
    """Memilih anime dari daftar populer"""
    # Hitung popularitas berdasarkan jumlah rating
    anime_popularity = recommender.rating_df['anime_id'].value_counts().head(20)
    popular_anime = recommender.anime_df[
        recommender.anime_df['anime_id'].isin(anime_popularity.index)
    ].copy()
    popular_anime['rating_count'] = popular_anime['anime_id'].map(anime_popularity)
    popular_anime = popular_anime.sort_values('rating_count', ascending=False)
    
    print(f"\n📈 TOP 20 ANIME POPULER:")
    print("-" * 80)
    for i, (_, anime) in enumerate(popular_anime.iterrows(), 1):
        print(f"{i:2d}. {anime['name']}")
        print(f"     Genre: {anime['genre']}")
        print(f"     Ratings: {anime['rating_count']} | Score: {anime['rating']}")
        print()
    
    try:
        choice = int(input(f"Pilih anime (1-{len(popular_anime)}): ")) - 1
        if 0 <= choice < len(popular_anime):
            selected_anime = popular_anime.iloc[choice]['name']
            get_content_recommendations_interactive(recommender, selected_anime)
        else:
            print("❌ Pilihan tidak valid!")
    except ValueError:
        print("❌ Input harus berupa angka!")

def select_from_active_users(recommender):
    """Memilih user dari daftar user aktif"""
    user_activity = recommender.rating_df['user_id'].value_counts().head(20)
    
    print(f"\n👥 TOP 20 USER PALING AKTIF:")
    print("-" * 50)
    for i, (user_id, count) in enumerate(user_activity.items(), 1):
        print(f"{i:2d}. User {user_id}: {count} ratings")
    
    try:
        choice = int(input(f"Pilih user (1-{len(user_activity)}): ")) - 1
        if 0 <= choice < len(user_activity):
            selected_user = user_activity.index[choice]
            get_collaborative_recommendations_interactive(recommender, selected_user)
        else:
            print("❌ Pilihan tidak valid!")
    except ValueError:
        print("❌ Input harus berupa angka!")

def show_popular_anime(recommender):
    """Menampilkan anime populer"""
    anime_popularity = recommender.rating_df['anime_id'].value_counts().head(15)
    popular_anime = recommender.anime_df[
        recommender.anime_df['anime_id'].isin(anime_popularity.index)
    ].copy()
    popular_anime['rating_count'] = popular_anime['anime_id'].map(anime_popularity)
    popular_anime = popular_anime.sort_values('rating_count', ascending=False)
    
    print(f"\n📈 TOP 15 ANIME POPULER:")
    print("=" * 80)
    for i, (_, anime) in enumerate(popular_anime.iterrows(), 1):
        print(f"{i:2d}. {anime['name']}")
        print(f"     Genre: {anime['genre']}")
        print(f"     Type: {anime['type']} | Rating: {anime['rating']}")
        print(f"     Total Ratings: {anime['rating_count']}")
        print()

def show_active_users(recommender):
    """Menampilkan user paling aktif"""
    user_activity = recommender.rating_df['user_id'].value_counts().head(15)
    
    print(f"\n👥 TOP 15 USER PALING AKTIF:")
    print("=" * 50)
    for i, (user_id, count) in enumerate(user_activity.items(), 1):
        user_data = recommender.rating_df[recommender.rating_df['user_id'] == user_id]
        avg_rating = user_data['rating'].mean()
        print(f"{i:2d}. User {user_id}")
        print(f"     Total Ratings: {count}")
        print(f"     Average Rating: {avg_rating:.2f}")
        print()

def search_anime(recommender):
    """Mencari anime berdasarkan kata kunci"""
    keyword = input("\n🔍 Masukkan kata kunci pencarian: ").strip()
    
    if not keyword:
        print("❌ Kata kunci tidak boleh kosong!")
        return
    
    # Cari berdasarkan nama
    name_matches = recommender.anime_df[
        recommender.anime_df['name'].str.contains(keyword, case=False, na=False)
    ]
    
    # Cari berdasarkan genre
    genre_matches = recommender.anime_df[
        recommender.anime_df['genre'].str.contains(keyword, case=False, na=False)
    ]
    
    # Gabungkan hasil
    all_matches = pd.concat([name_matches, genre_matches]).drop_duplicates()
    
    if all_matches.empty:
        print(f"❌ Tidak ditemukan anime dengan kata kunci '{keyword}'")
        return
    
    print(f"\n🔍 HASIL PENCARIAN UNTUK '{keyword}':")
    print(f"Ditemukan {len(all_matches)} anime:")
    print("=" * 80)
    
    for i, (_, anime) in enumerate(all_matches.head(20).iterrows(), 1):
        print(f"{i:2d}. {anime['name']}")
        print(f"     Genre: {anime['genre']}")
        print(f"     Type: {anime['type']} | Rating: {anime['rating']}")
        print()
    
    if len(all_matches) > 20:
        print(f"... dan {len(all_matches) - 20} anime lainnya")

def evaluate_model_menu(recommender):
    """Menu evaluasi model"""
    print(f"\n📈 EVALUASI MODEL:")
    print("=" * 50)
    
    try:
        rmse = recommender.evaluate_model()
        
        print(f"✅ RMSE Score: {rmse:.4f}")
        
        if rmse < 1.0:
            print("🎉 Performa Excellent! Model sangat akurat.")
        elif rmse < 2.0:
            print("👍 Performa Good! Model cukup akurat.")
        elif rmse < 3.0:
            print("⚠️ Performa Average. Model masih bisa diperbaiki.")
        else:
            print("⚠️ Performa perlu ditingkatkan.")
        
        print(f"\n📊 STATISTIK DATASET:")
        print(f"   • Total Anime: {len(recommender.anime_df):,}")
        print(f"   • Total Users: {recommender.rating_df['user_id'].nunique():,}")
        print(f"   • Total Ratings: {len(recommender.rating_df):,}")
        print(f"   • Average Rating: {recommender.rating_df['rating'].mean():.2f}")
        
    except Exception as e:
        print(f"❌ Error dalam evaluasi: {str(e)}")

if __name__ == "__main__":
    try:
        interactive_recommendation_system()
    except KeyboardInterrupt:
        print("\n\n🙏 Program dihentikan oleh user. Terima kasih!")
    except Exception as e:
        print(f"\n❌ Terjadi error: {str(e)}")
        print("Please check your dataset files and try again.")