import requests
import face_recognition
import concurrent.futures
import os
import time
import re
import random
import json
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urljoin
from PIL import Image, ImageEnhance
from io import BytesIO
from fake_useragent import UserAgent

class PersonSearcher:
    def __init__(self):
        self.ua = UserAgent()
        self.search_results = {
            'name': None,
            'verified_name': None,
            'gender': None,
            'email': None,
            'phone': None,
            'birth_info': None,
            'origin': None,
            'social_media': {},
            'other_info': {},
            'matched_faces': []
        }
        self.social_platforms = [
            'facebook.com', 'instagram.com', 'linkedin.com', 
            'twitter.com', 'tiktok.com', 'youtube.com'
        ]
        self.search_engines = ['google', 'duckduckgo']
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

    def get_random_proxy(self):
        proxies = [
            None,  # Sometimes use direct connection
            {'http': 'http://127.0.0.1:8080', 'https': 'http://127.0.0.1:8080'},
            {'http': 'http://127.0.0.1:8118', 'https': 'http://127.0.0.1:8118'},
        ]
        return random.choice(proxies)

    def load_face_image(self, image_path):
        try:
            if not os.path.exists(image_path):
                print(f"Error: Image file not found at {image_path}")
                return None
            
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                print("No faces detected in the image. Attempting image enhancement...")
                enhanced_image = self.enhance_image(image_path)
                face_encodings = face_recognition.face_encodings(enhanced_image)
                
                if not face_encodings:
                    print("Still no faces detected after enhancement. Please try with a clearer image.")
                    return None
                
            return face_encodings[0]  # Return the first face encoding
        except Exception as e:
            print(f"Error loading face image: {e}")
            return None

    def enhance_image(self, image_path):
        try:
            img = Image.open(image_path)
            
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.0)
            
            enhanced_path = f"enhanced_{os.path.basename(image_path)}"
            img.save(enhanced_path)
            
            return face_recognition.load_image_file(enhanced_path)
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return face_recognition.load_image_file(image_path)

    def search_by_name(self, name, search_engine, worker_id):
        if search_engine == 'google':
            return self.search_google(name, worker_id)
        elif search_engine == 'duckduckgo':
            return self.search_duckduckgo(name, worker_id)
        return []

    def search_google(self, name, worker_id):
        print(f"Worker {worker_id}: Searching Google for {name}")
        query = quote_plus(name)
        urls = []
        
        try:
            search_url = f"https://www.google.com/search?q={query}"
            response = self.session.get(
                search_url, 
                headers={'User-Agent': self.ua.random},
                proxies=self.get_random_proxy(),
                timeout=10
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for link in soup.select('a[href]'):
                    href = link.get('href', '')
                    if href.startswith('/url?q='):
                        url = href.split('/url?q=')[1].split('&')[0]
                        if any(platform in url for platform in self.social_platforms):
                            urls.append(url)
                
                for result in soup.select('div.g'):
                    result_text = result.get_text().lower()
                    self.extract_personal_info(result_text, name)
                    
                image_search_url = f"https://www.google.com/search?q={query}&tbm=isch"
                img_response = self.session.get(
                    image_search_url,
                    headers={'User-Agent': self.ua.random},
                    proxies=self.get_random_proxy(),
                    timeout=10
                )
                
                if img_response.status_code == 200:
                    soup = BeautifulSoup(img_response.text, 'html.parser')
                    img_elements = soup.select('img')
                    
                    for img in img_elements[1:10]:  # Skip Google logo, take next 9
                        src = img.get('src', '')
                        if src.startswith('http') and 'gif' not in src.lower():
                            urls.append(src)
            
            time.sleep(random.uniform(1, 3))  # Avoid rate limiting
        except Exception as e:
            print(f"Worker {worker_id}: Error searching Google: {e}")
        
        return urls

    def search_duckduckgo(self, name, worker_id):
        print(f"Worker {worker_id}: Searching DuckDuckGo for {name}")
        query = quote_plus(name)
        urls = []
        
        try:
            search_url = f"https://duckduckgo.com/html/?q={query}"
            response = self.session.get(
                search_url,
                headers={'User-Agent': self.ua.random},
                proxies=self.get_random_proxy(),
                timeout=10
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for link in soup.select('a.result__a'):
                    href = link.get('href', '')
                    if any(platform in href for platform in self.social_platforms):
                        urls.append(href)
                
                for result in soup.select('div.result'):
                    result_text = result.get_text().lower()
                    self.extract_personal_info(result_text, name)
            
            time.sleep(random.uniform(1, 3))  # Avoid rate limiting
        except Exception as e:
            print(f"Worker {worker_id}: Error searching DuckDuckGo: {e}")
        
        return urls

    def search_social_media(self, name, platform, worker_id):
        print(f"Worker {worker_id}: Searching {platform} for {name}")
        results = []
        
        try:
            if platform == 'facebook.com':
                search_url = f"https://www.facebook.com/public/{name.replace(' ', '-')}"
            elif platform == 'instagram.com':
                search_url = f"https://www.instagram.com/explore/tags/{name.replace(' ', '')}"
            elif platform == 'linkedin.com':
                search_url = f"https://www.linkedin.com/pub/dir?firstName={name.split()[0]}&lastName={name.split()[-1]}"
            elif platform == 'twitter.com':
                search_url = f"https://twitter.com/search?q={quote_plus(name)}&src=typed_query"
            elif platform == 'tiktok.com':
                search_url = f"https://www.tiktok.com/search?q={quote_plus(name)}"
            elif platform == 'youtube.com':
                search_url = f"https://www.youtube.com/results?search_query={quote_plus(name)}"
            else:
                return results
            
            response = self.session.get(
                search_url,
                headers={'User-Agent': self.ua.random},
                proxies=self.get_random_proxy(),
                timeout=15
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                if platform == 'facebook.com':
                    profiles = soup.select('div.x1qlqyl8')
                    for profile in profiles[:3]:
                        name_elem = profile.select_one('span')
                        if name_elem:
                            profile_name = name_elem.get_text().strip()
                            profile_url = profile.select_one('a')
                            if profile_url:
                                results.append({
                                    'platform': 'Facebook',
                                    'name': profile_name,
                                    'url': urljoin('https://facebook.com', profile_url.get('href', ''))
                                })
                
                elif platform == 'instagram.com':
                    profiles = soup.select('div._aagu')
                    for profile in profiles[:3]:
                        img = profile.select_one('img')
                        if img:
                            img_url = img.get('src', '')
                            if img_url:
                                results.append({
                                    'platform': 'Instagram',
                                    'image_url': img_url
                                })
                
                elif platform == 'linkedin.com':
                    profiles = soup.select('li.reusable-search__result-container')
                    for profile in profiles[:3]:
                        name_elem = profile.select_one('span.entity-result__title-text')
                        if name_elem:
                            profile_name = name_elem.get_text().strip()
                            profile_url = profile.select_one('a.app-aware-link')
                            if profile_url:
                                results.append({
                                    'platform': 'LinkedIn',
                                    'name': profile_name,
                                    'url': profile_url.get('href', '')
                                })
                
                elif platform == 'twitter.com':
                    profiles = soup.select('div[data-testid="cellInnerDiv"]')
                    for profile in profiles[:3]:
                        name_elem = profile.select_one('div[data-testid="User-Name"]')
                        if name_elem:
                            profile_name = name_elem.get_text().strip()
                            results.append({
                                'platform': 'Twitter',
                                'name': profile_name
                            })
                
                elif platform in ['tiktok.com', 'youtube.com']:
                    img_elements = soup.select('img')
                    for img in img_elements[:5]:
                        img_url = img.get('src', '')
                        if img_url and img_url.startswith('http'):
                            results.append({
                                'platform': 'TikTok' if platform == 'tiktok.com' else 'YouTube',
                                'image_url': img_url
                            })
            
            time.sleep(random.uniform(2, 4))  # Avoid rate limiting
        except Exception as e:
            print(f"Worker {worker_id}: Error searching {platform}: {e}")
        
        return results

    def extract_personal_info(self, text, name):
        text = text.lower()
        name_parts = name.split()
        
        if not self.search_results['verified_name']:
            name_pattern = r'(?:name|full name|real name)[:\s]+([a-zA-Z\s]+)'
            name_match = re.search(name_pattern, text)
            if name_match:
                potential_name = name_match.group(1).strip().title()
                if any(part in potential_name.lower() for part in name_parts):
                    self.search_results['verified_name'] = potential_name
        
        if not self.search_results['gender']:
            if re.search(r'\b(he|his|him|male|man|boy|gentleman)\b', text):
                self.search_results['gender'] = 'Male'
            elif re.search(r'\b(she|her|hers|female|woman|girl|lady)\b', text):
                self.search_results['gender'] = 'Female'
        
        if not self.search_results['email']:
            email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
            email_match = re.search(email_pattern, text)
            if email_match:
                potential_email = email_match.group(0)
                if any(part in potential_email for part in name_parts):
                    self.search_results['email'] = potential_email
        
        if not self.search_results['phone']:
            phone_pattern = r'(\+\d{1,3}[-\.\s]?)?(\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4})'
            phone_match = re.search(phone_pattern, text)
            if phone_match:
                self.search_results['phone'] = phone_match.group(0)
        
        if not self.search_results['birth_info']:
            birth_patterns = [
                r'born (?:on|in)? (\w+ \d{1,2},? \d{4})',
                r'born (?:on|in)? (\d{1,2} \w+ \d{4})',
                r'born (?:on|in)? (\w+,? \d{4})',
                r'born (?:on|in)? (\d{4})',
                r'birthdate:? (\w+ \d{1,2},? \d{4})',
                r'birth date:? (\w+ \d{1,2},? \d{4})',
                r'date of birth:? (\w+ \d{1,2},? \d{4})'
            ]
            
            for pattern in birth_patterns:
                birth_match = re.search(pattern, text)
                if birth_match:
                    self.search_results['birth_info'] = birth_match.group(1)
                    break
        
        if not self.search_results['origin']:
            origin_patterns = [
                r'(?:born|from|originates?) (?:in|from) ([A-Za-z\s]+)',
                r'nationality:? ([A-Za-z\s]+)',
                r'origin:? ([A-Za-z\s]+)',
                r'citizen of ([A-Za-z\s]+)'
            ]
            
            for pattern in origin_patterns:
                origin_match = re.search(pattern, text)
                if origin_match:
                    origin = origin_match.group(1).strip()
                    if len(origin) > 2 and origin.lower() not in ['in', 'on', 'at', 'the']:
                        self.search_results['origin'] = origin.title()
                        break

    def compare_face_with_url(self, face_encoding, image_url, worker_id):
        try:
            response = self.session.get(
                image_url, 
                headers={'User-Agent': self.ua.random},
                stream=True,
                timeout=10
            )
            
            if response.status_code == 200:
                try:
                    img = Image.open(BytesIO(response.content))
                    img_path = f"temp_face_{worker_id}_{int(time.time())}.jpg"
                    img.save(img_path)
                    
                    unknown_image = face_recognition.load_image_file(img_path)
                    unknown_face_encodings = face_recognition.face_encodings(unknown_image)
                    
                    if unknown_face_encodings:
                        matches = face_recognition.compare_faces([face_encoding], unknown_face_encodings[0], tolerance=0.6)
                        if matches[0]:
                            face_distance = face_recognition.face_distance([face_encoding], unknown_face_encodings[0])[0]
                            confidence = 1 - face_distance
                            
                            if confidence > 0.5:
                                result = {
                                    'url': image_url,
                                    'confidence': float(confidence),
                                    'match': True
                                }
                                self.search_results['matched_faces'].append(result)
                                print(f"Worker {worker_id}: Face match found! Confidence: {confidence:.2f}")
                                return result
                    
                    os.remove(img_path)
                except Exception as e:
                    print(f"Worker {worker_id}: Error processing image: {e}")
        except Exception as e:
            print(f"Worker {worker_id}: Error downloading image: {e}")
        
        return None

    def worker_task(self, worker_id, name, face_encoding, search_type):
        results = []
        
        if search_type == 'name_search':
            engine = self.search_engines[worker_id % len(self.search_engines)]
            urls = self.search_by_name(name, engine, worker_id)
            results.extend(urls)
            
        elif search_type == 'social_search':
            platform_index = worker_id % len(self.social_platforms)
            platform = self.social_platforms[platform_index]
            social_results = self.search_social_media(name, platform, worker_id)
            
            for result in social_results:
                if 'url' in result:
                    results.append(result['url'])
                if 'image_url' in result:
                    results.append(result['image_url'])
                
                if 'platform' in result and 'name' in result:
                    if result['platform'] not in self.search_results['social_media']:
                        self.search_results['social_media'][result['platform']] = []
                    
                    self.search_results['social_media'][result['platform']].append({
                        'name': result.get('name', ''),
                        'url': result.get('url', '')
                    })
        
        elif search_type == 'face_search' and face_encoding is not None:
            for url in results:
                if url.startswith('http') and any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                    self.compare_face_with_url(face_encoding, url, worker_id)
        
        return results

    def search_person(self, name, image_path):
        self.search_results['name'] = name
        face_encoding = self.load_face_image(image_path)
        all_urls = []
        
        print(f"Starting search for {name} with image: {image_path}")
        
        if face_encoding is None:
            print("Warning: Unable to detect a face in the provided image.")
            print("Continuing search based on name only...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            thread_futures = []
            
            for thread_id in range(2):
                future = executor.submit(
                    self.process_thread_tasks, thread_id, name, face_encoding
                )
                thread_futures.append(future)
            
            for future in concurrent.futures.as_completed(thread_futures):
                urls = future.result()
                all_urls.extend(urls)
        
        if face_encoding is not None and all_urls:
            print("Performing facial recognition on discovered images...")
            image_urls = [url for url in all_urls if url.startswith('http') and any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp'])]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as face_executor:
                face_futures = []
                
                for i, url in enumerate(image_urls):
                    future = face_executor.submit(
                        self.compare_face_with_url, face_encoding, url, i
                    )
                    face_futures.append(future)
                
                concurrent.futures.wait(face_futures)
        
        return self.search_results

    def process_thread_tasks(self, thread_id, name, face_encoding):
        thread_urls = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as worker_executor:
            worker_futures = []
            
            for worker_id in range(3):
                global_worker_id = thread_id * 3 + worker_id
                
                if global_worker_id < 3:
                    search_type = 'name_search'
                elif global_worker_id < 6:
                    search_type = 'social_search'
                
                future = worker_executor.submit(
                    self.worker_task, global_worker_id, name, face_encoding, search_type
                )
                worker_futures.append(future)
            
            for future in concurrent.futures.as_completed(worker_futures):
                urls = future.result()
                thread_urls.extend(urls)
        
        return thread_urls

    def process_results(self):
        if not self.search_results['verified_name']:
            self.search_results['verified_name'] = self.search_results['name'].title()
            
        self.search_results['matched_faces'].sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        if self.search_results['matched_faces']:
            print(f"\nFound {len(self.search_results['matched_faces'])} face matches")
            best_match = self.search_results['matched_faces'][0]
            print(f"Best match confidence: {best_match['confidence']:.2f}")
        else:
            print("\nNo face matches found")
            
        print("\nInformation found:")
        for key, value in self.search_results.items():
            if key != 'matched_faces' and value:
                if key == 'social_media':
                    print(f"- Social Media Profiles:")
                    for platform, profiles in value.items():
                        print(f"  * {platform}: {len(profiles)} potential profiles")
                else:
                    print(f"- {key.replace('_', ' ').title()}: {value}")
        
        return self.search_results


def main():
    searcher = PersonSearcher()
    
    print("=" * 60)
    print("   Person Information Search Tool using Face Recognition")
    print("=" * 60)
    
    name = input("\nEnter the person's full name (all lowercase): ").strip().lower()
    image_path = input("\nEnter the path to the face image (in current folder): ").strip()
    
    if not image_path.startswith('/'):
        image_path = f"./{image_path}"
    
    print("\nStarting search with 6 workers and 2 threads...")
    print("This may take several minutes. Please wait...\n")
    
    start_time = time.time()
    results = searcher.search_person(name, image_path)
    searcher.process_results()
    
    output_file = f"{name.replace(' ', '_')}_search_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    elapsed_time = time.time() - start_time
    print(f"\nSearch completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
