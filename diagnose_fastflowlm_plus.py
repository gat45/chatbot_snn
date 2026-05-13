#!/usr/bin/env python3
"""
Script de diagnostic pour la communication avec FastFlowLM
Teste la connectivité, les modèles disponibles, et les chat completions
"""

import asyncio
import httpx
import os
import sys
import json
from datetime import datetime

# Configuration
FASTFLOWLM_BASE = os.getenv("FASTFLOWLM_BASE", "http://127.0.0.1:52625/v1")

async def test_connectivity(client: httpx.AsyncClient) -> bool:
    """Test 1: Vérifier la connectivité de base"""
    print("Test 1: Connectivité réseau")
    print("-" * 60)
    
    try:
        response = await client.get(f"{FASTFLOWLM_BASE}/models", timeout=5.0)
        print(f"✅ Connexion réussie!")
        print(f"   Status: {response.status_code}")
        print(f"   URL: {FASTFLOWLM_BASE}/models")
        
        if response.status_code == 200:
            return True
        else:
            print(f"   ⚠️  Status code inattendu: {response.status_code}")
            print(f"   Body: {response.text[:200]}")
            return False
            
    except httpx.ConnectError as e:
        print(f"❌ Impossible de se connecter")
        print(f"   Erreur: {e}")
        print(f"   → Vérifier que FastFlowLM est lancé sur {FASTFLOWLM_BASE}")
        return False
        
    except httpx.TimeoutException as e:
        print(f"❌ Timeout après 5 secondes")
        print(f"   Erreur: {e}")
        print(f"   → FastFlowLM est trop lent ou bloqué")
        return False
        
    except Exception as e:
        print(f"❌ Erreur inattendue: {type(e).__name__}")
        print(f"   Message: {e}")
        return False

async def test_models(client: httpx.AsyncClient) -> list:
    """Test 2: Lister les modèles disponibles"""
    print("\nTest 2: Liste des modèles disponibles")
    print("-" * 60)
    
    try:
        response = await client.get(f"{FASTFLOWLM_BASE}/models", timeout=5.0)
        
        if response.status_code != 200:
            print(f"❌ Erreur {response.status_code}")
            print(f"   Body: {response.text}")
            return []
        
        data = response.json()
        models = data.get("data", [])
        
        if not models:
            print("⚠️  Aucun modèle disponible")
            print("   → Vérifier que FastFlowLM a chargé au moins un modèle")
            return []
        
        print(f"✅ {len(models)} modèle(s) trouvé(s):")
        for i, model in enumerate(models, 1):
            model_id = model.get("id", "N/A")
            model_type = model.get("object", "N/A")
            print(f"   {i}. {model_id} (type: {model_type})")
        
        return models
        
    except json.JSONDecodeError as e:
        print(f"❌ Réponse JSON invalide")
        print(f"   Erreur: {e}")
        print(f"   Body: {response.text[:200]}")
        return []
        
    except Exception as e:
        print(f"❌ Erreur: {type(e).__name__}: {e}")
        return []

async def get_loaded_model_info(client: httpx.AsyncClient) -> str:
    """Récupérer le modèle actuellement chargé via une requête de test"""
    print(f"\nRécupération du modèle chargé...")

    # Envoyer une requête de test pour obtenir des informations sur le modèle chargé
    payload = {
        "model": "gemma3:1b",  # Utiliser le modèle connu pour obtenir des infos
        "messages": [
            {"role": "user", "content": "Dis juste 'OK'."}
        ],
        "temperature": 0.0,
        "max_tokens": 5,
        "stream": False
    }

    try:
        response = await client.post(
            f"{FASTFLOWLM_BASE}/chat/completions",
            json=payload,
            timeout=10.0
        )

        if response.status_code == 200:
            result = response.json()

            # Le modèle réellement utilisé est souvent inclus dans la réponse
            loaded_model = result.get("model", "")
            if loaded_model:
                print(f"   → Modèle chargé détecté: '{loaded_model}'")
                return loaded_model
            else:
                print(f"   → Aucune information sur le modèle chargé trouvée dans la réponse")
                return ""
        else:
            print(f"   → Impossible d'obtenir les infos du modèle chargé: {response.status_code}")
            return ""

    except Exception as e:
        print(f"   → Erreur lors de la récupération des infos du modèle: {e}")
        return ""


async def test_model_availability(client: httpx.AsyncClient, model_id: str) -> bool:
    """Tester si un modèle spécifique est réellement disponible et fonctionnel"""
    print(f"\nTest de disponibilité du modèle: '{model_id}'")

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Dis juste 'OK' pour confirmer que tu fonctionnes."}
        ],
        "temperature": 0.0,  # Température basse pour une réponse rapide et déterministe
        "max_tokens": 5,     # Moins de tokens pour un test rapide
        "stream": False
    }

    try:
        response = await client.post(
            f"{FASTFLOWLM_BASE}/chat/completions",
            json=payload,
            timeout=15.0  # Délai plus court pour les tests
        )

        if response.status_code == 200:
            result = response.json()

            # Vérifier la structure de base de la réponse
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"].strip()

                    # Vérifier que la réponse est significative et pas juste des caractères répétés
                    if content and len(content) > 0 and not is_repetitive_response(content):
                        # Vérifier si le modèle réellement utilisé correspond à celui demandé
                        actual_model = result.get("model", "")
                        if actual_model and actual_model != model_id:
                            print(f"   ⚠️  Modèle demandé: '{model_id}', mais modèle réel: '{actual_model}'")
                            print(f"   ✅ Modèle '{actual_model}' est fonctionnel - Réponse: '{content[:30]}...'")
                        else:
                            print(f"   ✅ Modèle '{model_id}' est fonctionnel - Réponse: '{content[:30]}...'")

                        return True
                    else:
                        print(f"   ❌ Modèle '{model_id}' - Réponse vide ou répétitive: '{content[:30]}...'")
                else:
                    print(f"   ❌ Modèle '{model_id}' - Structure de réponse incomplète")
            else:
                print(f"   ❌ Modèle '{model_id}' - Réponse sans choix ('choices')")
        else:
            # Vérifier si c'est une erreur de téléchargement de modèle
            if "missing" in response.text.lower() or "download" in response.text.lower():
                print(f"   ❌ Modèle '{model_id}' - Nécessite un téléchargement (non chargé)")
            else:
                print(f"   ❌ Modèle '{model_id}' - Erreur {response.status_code}: {response.text[:100]}")
        return False

    except Exception as e:
        error_msg = str(e).lower()
        if "missing" in error_msg or "download" in error_msg or "not found" in error_msg:
            print(f"   ❌ Modèle '{model_id}' - Nécessite un téléchargement (non chargé)")
        else:
            print(f"   ❌ Modèle '{model_id}' - Erreur: {type(e).__name__}: {str(e)[:100]}")
        return False


def is_repetitive_response(text: str) -> bool:
    """Vérifier si la réponse est composée de caractères répétitifs"""
    if len(text) < 3:
        return False

    # Vérifier si plus de 70% des caractères sont identiques
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1

    most_common_char_count = max(char_counts.values()) if char_counts else 0
    return most_common_char_count / len(text) > 0.7


async def run_performance_benchmark(client: httpx.AsyncClient, model_id: str) -> dict:
    """Exécuter un benchmark de performance sur le modèle"""
    print(f"\nBenchmark de performance pour: '{model_id}'")

    benchmark_results = {
        "avg_response_time": 0,
        "min_response_time": float('inf'),
        "max_response_time": 0,
        "throughput": 0,  # Tokens par seconde
        "latency_percentiles": {},
        "num_requests": 0,
        "successful_requests": 0
    }

    # Paramètres du benchmark
    num_tests = 5
    test_payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Explique brièvement ce qu'est l'intelligence artificielle."}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    response_times = []

    for i in range(num_tests):
        print(f"  Test {i+1}/{num_tests}...")
        start_time = asyncio.get_event_loop().time()

        try:
            response = await client.post(
                f"{FASTFLOWLM_BASE}/chat/completions",
                json=test_payload,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()

                # Calculer le temps de réponse
                end_time = asyncio.get_event_loop().time()
                response_time = end_time - start_time
                response_times.append(response_time)

                # Calculer le débit (tokens/seconde) si possible
                if "usage" in result:
                    usage = result["usage"]
                    total_tokens = usage.get("total_tokens", 0)
                    if total_tokens > 0 and response_time > 0:
                        throughput = total_tokens / response_time
                        if throughput > benchmark_results["throughput"]:
                            benchmark_results["throughput"] = throughput

                benchmark_results["successful_requests"] += 1
            else:
                print(f"    → Échec du test {i+1}: {response.status_code}")

        except Exception as e:
            print(f"    → Erreur lors du test {i+1}: {str(e)[:100]}")

    benchmark_results["num_requests"] = num_tests

    if response_times:
        benchmark_results["avg_response_time"] = round(sum(response_times) / len(response_times), 3)
        benchmark_results["min_response_time"] = round(min(response_times), 3)
        benchmark_results["max_response_time"] = round(max(response_times), 3)

        # Calculer les percentiles
        sorted_times = sorted(response_times)
        n = len(sorted_times)
        if n >= 2:
            p50_idx = n // 2
            p90_idx = int(0.9 * n)
            p95_idx = int(0.95 * n)

            benchmark_results["latency_percentiles"] = {
                "p50": round(sorted_times[p50_idx], 3),
                "p90": round(sorted_times[min(p90_idx, n-1)], 3),
                "p95": round(sorted_times[min(p95_idx, n-1)], 3)
            }

    print(f"  → Résultats du benchmark:")
    print(f"    • Réponses réussies: {benchmark_results['successful_requests']}/{num_tests}")
    print(f"    • Temps moyen: {benchmark_results['avg_response_time']}s")
    print(f"    • Temps min: {benchmark_results['min_response_time']}s")
    print(f"    • Temps max: {benchmark_results['max_response_time']}s")
    print(f"    • Débit max: {round(benchmark_results['throughput'], 2)} tokens/s")
    if benchmark_results["latency_percentiles"]:
        p95 = benchmark_results["latency_percentiles"]["p95"]
        print(f"    • Latence p95: {p95}s")

    return benchmark_results


async def test_model_capabilities(client: httpx.AsyncClient, model_id: str) -> dict:
    """Tester les capacités spécifiques du modèle"""
    print(f"\nTest des capacités du modèle: '{model_id}'")

    capabilities = {
        "basic_chat": False,
        "streaming": False,
        "function_calling": False,
        "response_time": None,
        "performance_benchmark": {}
    }

    # Test de base
    start_time = asyncio.get_event_loop().time()
    basic_test = await test_model_availability(client, model_id)
    end_time = asyncio.get_event_loop().time()

    capabilities["basic_chat"] = basic_test
    capabilities["response_time"] = round(end_time - start_time, 2)

    # Test du streaming si le modèle est fonctionnel
    if basic_test:
        try:
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": "Count from 1 to 5."}
                ],
                "temperature": 0.0,
                "max_tokens": 10,
                "stream": True
            }

            response = await client.post(
                f"{FASTFLOWLM_BASE}/chat/completions",
                json=payload,
                timeout=15.0
            )

            if response.status_code == 200:
                capabilities["streaming"] = True
                print(f"   ✅ Streaming supporté")
            else:
                print(f"   ❌ Streaming non supporté")

        except Exception as e:
            print(f"   ❌ Streaming non supporté ou erreur: {str(e)[:100]}")

    # Exécuter un benchmark de performance si le modèle est fonctionnel
    if basic_test:
        capabilities["performance_benchmark"] = await run_performance_benchmark(client, model_id)

    print(f"   → Temps de réponse: {capabilities['response_time']}s")
    print(f"   → Chat de base: {'✅' if capabilities['basic_chat'] else '❌'}")
    print(f"   → Streaming: {'✅' if capabilities['streaming'] else '❌'}")

    return capabilities


async def test_system_resources() -> dict:
    """Tester les ressources système disponibles"""
    print(f"\nTest des ressources système...")

    resources = {
        "cpu_count": 0,
        "memory_total": 0,
        "memory_available": 0,
        "sufficient_for_small_models": False,
        "sufficient_for_medium_models": False,
        "sufficient_for_large_models": False
    }

    try:
        import psutil

        resources["cpu_count"] = psutil.cpu_count()

        memory = psutil.virtual_memory()
        resources["memory_total"] = round(memory.total / (1024**3), 2)  # En Go
        resources["memory_available"] = round(memory.available / (1024**3), 2)  # En Go

        # Estimation des besoins en mémoire pour différents types de modèles
        resources["sufficient_for_small_models"] = resources["memory_available"] >= 4  # 4GB pour petits modèles
        resources["sufficient_for_medium_models"] = resources["memory_available"] >= 8  # 8GB pour modèles moyens
        resources["sufficient_for_large_models"] = resources["memory_available"] >= 16  # 16GB pour grands modèles

        print(f"   → CPU Cores: {resources['cpu_count']}")
        print(f"   → Mémoire totale: {resources['memory_total']} GB")
        print(f"   → Mémoire disponible: {resources['memory_available']} GB")
        print(f"   → Suffisant pour petits modèles: {'✅' if resources['sufficient_for_small_models'] else '❌'}")
        print(f"   → Suffisant pour modèles moyens: {'✅' if resources['sufficient_for_medium_models'] else '❌'}")
        print(f"   → Suffisant pour grands modèles: {'✅' if resources['sufficient_for_large_models'] else '❌'}")

    except ImportError:
        print(f"   → psutil non installé - impossible d'évaluer les ressources système")
        print(f"   → Pour installer psutil: pip install psutil")

    return resources


async def comprehensive_model_test(client: httpx.AsyncClient, models: list) -> tuple:
    """Effectuer un test complet de tous les modèles disponibles"""
    print("\nTest complet des modèles disponibles...")
    print("-" * 60)

    results = []

    for model in models:
        model_id = model.get("id", "")
        if not model_id:
            continue

        print(f"\nTest détaillé pour: {model_id}")
        print("=" * 40)

        # Test des capacités
        capabilities = await test_model_capabilities(client, model_id)

        # Regrouper les résultats
        model_result = {
            "model_id": model_id,
            "capabilities": capabilities,
            "is_usable": capabilities["basic_chat"]  # Un modèle est utilisable s'il passe le test de base
        }

        results.append(model_result)

        if model_result["is_usable"]:
            print(f"✅ Modèle '{model_id}' est pleinement fonctionnel")
        else:
            print(f"❌ Modèle '{model_id}' n'est pas utilisable")

    # Trouver le meilleur modèle utilisable
    usable_models = [r for r in results if r["is_usable"]]

    if usable_models:
        # Trier par temps de réponse (le plus rapide en premier)
        fastest_model = min(usable_models, key=lambda x: x["capabilities"]["response_time"])
        print(f"\n🎯 Meilleur modèle disponible: '{fastest_model['model_id']}' (réponse en {fastest_model['capabilities']['response_time']}s)")
        return fastest_model["model_id"], results
    else:
        print("\n❌ Aucun modèle n'est pleinement fonctionnel")
        return "", results


async def find_working_model(client: httpx.AsyncClient, models: list) -> str:
    """Trouver un modèle fonctionnel parmi la liste des modèles disponibles"""
    print("\nRecherche d'un modèle fonctionnel...")
    print("-" * 60)

    for model in models:
        model_id = model.get("id", "")
        if not model_id:
            continue

        print(f"Test du modèle: {model_id}")
        is_available = await test_model_availability(client, model_id)

        if is_available:
            print(f"✅ Modèle fonctionnel trouvé: '{model_id}'")
            return model_id
        else:
            print(f"   → Passer au modèle suivant...")

    print("❌ Aucun modèle fonctionnel trouvé parmi la liste")
    return ""


async def test_chat_completion(client: httpx.AsyncClient, model_id: str) -> bool:
    """Test 3: Tester une completion de chat avec le modèle spécifié"""
    print(f"\nTest 3: Chat completion avec modèle '{model_id}'")
    print("-" * 60)

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Dis juste 'OK' pour confirmer que tu fonctionnes."}
        ],
        "temperature": 0.7,
        "max_tokens": 20
    }

    print(f"Envoi de la requête...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()

    try:
        response = await client.post(
            f"{FASTFLOWLM_BASE}/chat/completions",
            json=payload,
            timeout=30.0
        )

        print(f"Status: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ Erreur {response.status_code}")
            print(f"   Body: {response.text[:500]}")
            return False

        result = response.json()

        # Vérifier la structure de la réponse
        if "choices" not in result:
            print("❌ Réponse invalide: 'choices' manquant")
            print(f"   Body: {json.dumps(result, indent=2)}")
            return False

        if len(result["choices"]) == 0:
            print("❌ Réponse invalide: 'choices' vide")
            return False

        choice = result["choices"][0]
        if "message" not in choice:
            print("❌ Réponse invalide: 'message' manquant")
            return False

        message = choice["message"]
        content = message.get("content", "")

        print("✅ Réponse reçue avec succès!")
        print(f"   ID: {result.get('id', 'N/A')}")
        print(f"   Modèle: {result.get('model', 'N/A')}")
        print(f"   Réponse: {content}")

        # Vérifier usage si présent
        if "usage" in result:
            usage = result["usage"]
            print(f"   Tokens prompt: {usage.get('prompt_tokens', 'N/A')}")
            print(f"   Tokens completion: {usage.get('completion_tokens', 'N/A')}")
            print(f"   Tokens total: {usage.get('total_tokens', 'N/A')}")

        return True

    except httpx.TimeoutException:
        print("❌ Timeout après 30 secondes")
        print("   → La génération est trop lente")
        print("   → Essayer avec max_tokens plus petit ou timeout plus grand")
        return False

    except json.JSONDecodeError as e:
        print(f"❌ Réponse JSON invalide")
        print(f"   Erreur: {e}")
        print(f"   Body: {response.text[:500]}")
        return False

    except Exception as e:
        print(f"❌ Erreur: {type(e).__name__}")
        print(f"   Message: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_network() -> None:
    """Test 4: Vérifier la configuration réseau"""
    print("\nTest 4: Configuration réseau")
    print("-" * 60)
    
    # Extraire host et port de l'URL
    import re
    match = re.match(r'https?://([^:/]+):(\d+)', FASTFLOWLM_BASE)
    
    if not match:
        print("⚠️  URL mal formée, impossible d'analyser")
        return
    
    host = match.group(1)
    port = match.group(2)
    
    print(f"Host: {host}")
    print(f"Port: {port}")
    
    # Test de connectivité socket
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, int(port)))
        sock.close()
        
        if result == 0:
            print(f"✅ Port {port} est ouvert et accessible")
        else:
            print(f"❌ Port {port} est fermé ou inaccessible")
            print(f"   → Vérifier que FastFlowLM écoute sur ce port")
    except Exception as e:
        print(f"⚠️  Impossible de tester le port: {e}")

def print_summary(connectivity_ok: bool, models_found: bool, chat_ok: bool) -> None:
    """Afficher un résumé des tests"""
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    status_connectivity = "✅ OK" if connectivity_ok else "❌ ÉCHEC"
    status_models = "✅ OK" if models_found else "❌ ÉCHEC"
    status_chat = "✅ OK" if chat_ok else "❌ ÉCHEC"
    
    print(f"1. Connectivité réseau:    {status_connectivity}")
    print(f"2. Modèles disponibles:    {status_models}")
    print(f"3. Chat completion:        {status_chat}")
    print()
    
    if connectivity_ok and models_found and chat_ok:
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("   → FastFlowLM est opérationnel")
        print("   → Le chatbot devrait pouvoir communiquer")
    else:
        print("⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
        print("\nRECOMMANDATIONS:")
        
        if not connectivity_ok:
            print("   1. Vérifier que FastFlowLM est lancé")
            print("   2. Vérifier l'URL et le port (FASTFLOWLM_BASE)")
            print("   3. Vérifier le firewall")
        
        if connectivity_ok and not models_found:
            print("   1. Charger au moins un modèle dans FastFlowLM")
            print("   2. Vérifier la configuration FastFlowLM")
        
        if models_found and not chat_ok:
            print("   1. Vérifier le format de la requête")
            print("   2. Augmenter le timeout")
            print("   3. Consulter les logs FastFlowLM")
    
    print("=" * 60)

async def main():
    """Fonction principale de diagnostic"""
    print("🔍 DIAGNOSTIC COMPLET FASTFLOWLM")
    print("=" * 60)
    print(f"URL cible: {FASTFLOWLM_BASE}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Créer le client HTTP
    client = httpx.AsyncClient()

    try:
        # Test 1: Connectivité
        connectivity_ok = await test_connectivity(client)

        if not connectivity_ok:
            print("\n❌ Impossible de continuer sans connectivité")
            print_summary(False, False, False)
            return 1

        # Test 2: Modèles
        models = await test_models(client)
        models_found = len(models) > 0

        if not models_found:
            print("\n❌ Impossible de continuer sans modèles")
            print_summary(True, False, False)
            return 1

        # Récupérer le modèle actuellement chargé
        loaded_model = await get_loaded_model_info(client)

        if loaded_model:
            print(f"\n🎯 Modèle réellement chargé: '{loaded_model}'")

            # Vérifier si ce modèle est dans la liste des modèles disponibles
            loaded_model_in_list = any(m.get("id", "") == loaded_model for m in models)
            if not loaded_model_in_list:
                print(f"⚠️  Le modèle chargé '{loaded_model}' n'est pas dans la liste des modèles disponibles")
                # Ajouter le modèle chargé à la liste s'il n'y est pas
                models.append({"id": loaded_model, "object": "model"})
        else:
            print(f"\n⚠️  Impossible de déterminer le modèle chargé")
            # Dans ce cas, utiliser l'ancienne méthode
            working_model_id, model_results = await comprehensive_model_test(client, models)
            if not working_model_id:
                print("\n❌ Impossible de continuer sans modèle fonctionnel")
                print_summary(True, True, False)
                return 1

        # Test des ressources système
        system_resources = await test_system_resources()

        # Si on a réussi à identifier le modèle chargé, on peut faire un test ciblé
        if loaded_model:
            # Effectuer un test complet uniquement sur le modèle chargé
            model_results = []
            capabilities = await test_model_capabilities(client, loaded_model)

            model_result = {
                "model_id": loaded_model,
                "capabilities": capabilities,
                "is_usable": capabilities["basic_chat"]  # Un modèle est utilisable s'il passe le test de base
            }

            model_results.append(model_result)

            if model_result["is_usable"]:
                working_model_id = loaded_model
                print(f"\n✅ Le modèle '{loaded_model}' est pleinement fonctionnel")
            else:
                print(f"\n❌ Le modèle '{loaded_model}' n'est pas utilisable")
                # Si le modèle chargé n'est pas utilisable, tester les autres modèles
                working_model_id, additional_results = await comprehensive_model_test(client, models)
                model_results.extend(additional_results)

                if not working_model_id:
                    print("\n❌ Impossible de continuer sans modèle fonctionnel")
                    print_summary(True, True, False)
                    return 1
        else:
            # Ancienne méthode si on ne peut pas identifier le modèle chargé
            working_model_id, model_results = await comprehensive_model_test(client, models)

            if not working_model_id:
                print("\n❌ Impossible de continuer sans modèle fonctionnel")
                print_summary(True, True, False)
                return 1

        # Test 3: Chat completion avec le modèle fonctionnel trouvé
        chat_ok = await test_chat_completion(client, working_model_id)

        # Test 4: Réseau
        await test_network()

        # Résumé
        print_summary(connectivity_ok, models_found, chat_ok)

        # Affichage des résultats détaillés
        print("\n" + "=" * 60)
        print("RÉSULTATS DÉTAILLÉS DES MODÈLES")
        print("=" * 60)
        for result in model_results:
            model_id = result["model_id"]
            caps = result["capabilities"]
            is_usable = result["is_usable"]

            print(f"\nModèle: {model_id}")
            print(f"  Utilisable: {'✅' if is_usable else '❌'}")
            print(f"  Temps de réponse: {caps['response_time']}s")
            print(f"  Chat de base: {'✅' if caps['basic_chat'] else '❌'}")
            print(f"  Streaming: {'✅' if caps['streaming'] else '❌'}")

        return 0 if chat_ok else 1

    finally:
        await client.aclose()

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnostic interrompu par l'utilisateur")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
