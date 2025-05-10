import os
import requests
import bs4
import cv2
import PIL.Image
import concurrent.futures
import urllib.parse
import re
import io
import numpy as np

def analyze_face(image_path):
    try:
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            return False, "Image not found or could not be read for OpenCV"

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return False, "No face detected in the image"

        (x, y, w, h) = faces[0]
        face_roi_gray = gray[y:y+h, x:x+w]

        hist = cv2.calcHist([face_roi_gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_L1)

        try:
            img_pil = PIL.Image.open(image_path)
            img_pil = img_pil.convert('RGB')
            colors = img_pil.getcolors(maxcolors=256 * 256 * 256)
            dominant_color = max(colors, key=lambda x: x[0])[1] if colors else None
        except Exception:
            dominant_color = None

        return True, {
            "face_detected": True,
            "face_roi_gray": face_roi_gray,
            "gray_hist": hist,
            "dominant_color": dominant_color
        }

    except Exception as e:
        return False, str(e)

def compare_faces_hist(hist1, hist2):
    if hist1 is None or hist2 is None:
        return 0
    try:
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    except cv2.error:
        return 0

def download_and_analyze_image(image_url):
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img_bytes = response.content
        img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

        if img_cv is None:
            return None, None

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(img_cv, 1.1, 4)

        if len(faces) == 0:
            return None, None

        (x, y, w, h) = faces[0]
        face_roi_gray = img_cv[y:y+h, x:x+w]

        hist = cv2.calcHist([face_roi_gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_L1)

        return face_roi_gray, hist

    except requests.exceptions.RequestException:
        return None, None
    except Exception:
        return None, None

def find_image_urls_in_soup(soup):
    image_urls = []
    for img_tag in soup.find_all('img'):
        src = img_tag.get('src')
        if src:
            image_urls.append(src)
    return image_urls

def perform_search(platform, name, face_data=None):
    results = []
    image_urls_found = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    query = f'{name} personal information contact social media profiles'
    url = ''
    if platform == 'google':
        url = f'https://www.google.com/search?q={urllib.parse.quote(query)}'
    elif platform == 'duckduckgo':
        url = f'https://duckduckgo.com/html/?q={urllib.parse.quote(query)}'
    else:
        return [], []

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, 'html.parser')

        image_urls_found.extend(find_image_urls_in_soup(soup))

        if platform == 'google':
            for g in soup.find_all('div', class_='tF2CMy'):
                link_tag = g.find('a')
                if link_tag:
                    link = link_tag['href']
                    title_tag = g.find('h3')
                    title = title_tag.text if title_tag else 'N/A'
                    snippet_tag = g.find('div', class_='lEBKkf') or g.find('div', class_='VwiC3b')
                    snippet = snippet_tag.text if snippet_tag else 'No snippet available.'

                    results.append({
                        'source': 'Google',
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })
        elif platform == 'duckduckgo':
             for r in soup.select('#links .result'):
                title_tag = r.select_one('.result__a')
                if title_tag:
                    title = title_tag.text
                    link = title_tag['href']
                    snippet_tag = r.select_one('.result__snippet')
                    snippet = snippet_tag.text if snippet_tag else 'No snippet available.'

                    results.append({
                        'source': 'DuckDuckGo',
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })


    except requests.exceptions.RequestException:
        pass

    return results, image_urls_found

def extract_info_from_text(text):
    extracted = {}
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if emails:
        extracted['emails'] = list(set(emails))

    phones = re.findall(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
    if phones:
        extracted['phones'] = list(set(phones))

    social_media_patterns = {
        'facebook': r'(?:https?://)?(?:www\.)?facebook\.com/[A-Za-z0-9_.]+',
        'twitter': r'(?:https?://)?(?:www\.)?twitter\.com/[A-Za-z0-9_]+',
        'linkedin': r'(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+',
        'instagram': r'(?:https?://)?(?:www\.)?instagram\.com/[A-Za-z0-9_.]+',
    }
    extracted['social_media'] = {}
    for platform, pattern in social_media_patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            extracted['social_media'][platform] = list(set(matches))

    date_patterns = [
        r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b',
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s+\d{4}\b',
    ]
    extracted['possible_birth_dates'] = []
    for pattern in date_patterns:
        extracted['possible_birth_dates'].extend(re.findall(pattern, text))
    extracted['possible_birth_dates'] = list(set(extracted['possible_birth_dates']))

    location_patterns = [
        r'\b(?:born in|from)\s+([A-Z][a-z]+(?:,\s*[A-Z]{2})?(?:,\s*[A-Z][a-z]+)?)\b',
        r'\b(?:lives in|based in)\s+([A-Z][a-z]+(?:,\s*[A-Z]{2})?(?:,\s*[A-Z][a-z]+)?)\b',
    ]
    extracted['possible_locations'] = []
    for pattern in location_patterns:
         extracted['possible_locations'].extend(re.findall(pattern, text))
    extracted['possible_locations'] = [loc[0] if isinstance(loc, tuple) else loc for loc in extracted['possible_locations']]
    extracted['possible_locations'] = list(set(extracted['possible_locations']))

    gender_patterns = [
        r'\b(he|him|his)\b',
        r'\b(she|her|hers)\b',
    ]
    found_genders = set()
    for pattern in gender_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if match.lower() in ['he', 'him', 'his']:
                found_genders.add('Male')
            elif match.lower() in ['she', 'her', 'hers']:
                found_genders.add('Female')
    if found_genders:
        extracted['possible_gender_pronouns'] = list(found_genders)

    return extracted

if __name__ == '__main__':
    person_name = input("Enter full name (lowercase, support space): ").lower()
    image_filename = input("Enter the filename of the face image (must be in this folder): ")
    image_path = os.path.join(os.getcwd(), image_filename)

    face_analysis_success, face_analysis_result = analyze_face(image_path)

    if not face_analysis_success:
        print(f"Face analysis failed: {face_analysis_result}")
        face_data = None
    else:
        print("Face analysis successful.")
        face_data = face_analysis_result

    all_results = []
    all_image_urls = []

    print("\nInitiating concurrent search...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = {}
        for i in range(3):
            futures[executor.submit(perform_search, 'google', person_name, face_data)] = 'google'
        for i in range(3):
            futures[executor.submit(perform_search, 'duckduckgo', person_name, face_data)] = 'duckduckgo'

        for future in concurrent.futures.as_completed(futures):
            platform = futures[future]
            try:
                results, image_urls = future.result()
                all_results.extend(results)
                all_image_urls.extend(image_urls)
                print(f" {platform.capitalize()} search finished.")
            except Exception as exc:
                print(f" {platform.capitalize()} search generated an exception: {exc}")

    print("\nSearch Results Processing:")
    extracted_info = {}
    for res in all_results:
        if 'error' in res:
            continue
        info = extract_info_from_text(res['snippet'] + ' ' + res['title'] + ' ' + res.get('link', ''))
        for key, values in info.items():
            if key not in extracted_info:
                extracted_info[key] = set()
            for value in values:
                extracted_info[key].add(value)

    print("\nImage Correlation Attempt:")
    potential_image_matches = []
    if face_data and face_data["face_detected"]:
        input_hist = face_data["gray_hist"]
        unique_image_urls = list(set(all_image_urls))
        print(f" Attempting to download and analyze {len(unique_image_urls)} images found online using 2 threads.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as image_executor:
            image_analysis_futures = {image_executor.submit(download_and_analyze_image, url): url for url in unique_image_urls}
            for future in concurrent.futures.as_completed(image_analysis_futures):
                url = image_analysis_futures[future]
                try:
                    img_roi_gray, hist = future.result()
                    if hist is not None:
                        correlation = compare_faces_hist(input_hist, hist)
                        if correlation > 0.7:
                            potential_image_matches.append({'url': url, 'correlation': correlation})
                            print(f"  Potential match found (Correlation: {correlation:.2f}): {url}")
                except Exception:
                    pass

    print("\n--- OSINT Report ---")

    print("\nPotential Extracted Information (from text analysis):")
    if extracted_info:
        for key, values in extracted_info.items():
            print(f"{key.replace('_', ' ').title()}:")
            if values:
                for value in values:
                    print(f" - {value}")
            else:
                print(" - None found")
    else:
        print("No specific information patterns found in text results.")

    print("\nPotential Image Correlations (based on face histogram matching > 0.7):")
    if potential_image_matches:
        sorted_matches = sorted(potential_image_matches, key=lambda x: x['correlation'], reverse=True)
        for match in sorted_matches:
             print(f" - Correlation {match['correlation']:.2f}: {match['url']}")
    else:
        print("No strong potential image correlations found.")

    print("\nNote: Automated scraping and analysis provide potential leads. Verification of information (Real Name, Gender, exact Birthdate/Birthplace, Originality) requires manual review and cross-referencing of sources.")
