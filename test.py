# test_complete_100.py - Complete memory test with 100 items
import os
import random
import time
from core.generator import GeneratorManager
from Memory.Memory_manager import MemoryManager
from Memory.High_level_memory import LTM_index

def clear_all_memory():
    """Clear all memory files"""
    files = ["identity_index.faiss", "emotional_index.faiss", "semantic_index.faiss", 
             "episodic_index.faiss", "procedural_index.faiss", "code_index.faiss",
             "LTM_text.pkl", "memory_index.faiss", "memory_data.pkl"]
    deleted = 0
    for f in files:
        if os.path.exists(f):
            os.remove(f)
            deleted += 1
    print(f"🗑️ Deleted {deleted} memory files")
    return deleted

def print_result(test_num, test_name, expected, actual, status):
    """Print test result in formatted way"""
    symbol = "✅" if status else "❌"
    print(f"{symbol} Test {test_num:3d}: {test_name[:45]:45s} → {actual[:45] if actual else 'NOT FOUND'}")

def main():
    print("="*70)
    print("🧠 COMPLETE MEMORY TEST - 100 ITEMS")
    print("="*70)
    
    # Clear old memory
    clear_all_memory()
    
    # Initialize
    print("\n⏳ Loading models...")
    gen = GeneratorManager(r'C:\Users\itsN3Fi\Desktop\RAG_WITH_TWO_TYPE_MEMORY\Generator_config.json')
    memory = MemoryManager(STM_SIZE=20, gen=gen)
    memory.load_all()
    print("✅ Ready!\n")
    
    # ==================== PART 1: SAVE 100 MEMORIES ====================
    print("="*70)
    print("📝 PART 1: SAVING 100 MEMORIES")
    print("="*70)
    
    # ========== IDENTITY memories (20 items) ==========
    identity_data = [
        ("my name is John Smith", "identity"),
        ("i am 25 years old", "identity"),
        ("i live in New York", "identity"),
        ("i work as a software engineer", "identity"),
        ("my birthday is on May 15th", "identity"),
        ("my email is john@gmail.com", "identity"),
        ("my phone number is 555-0100", "identity"),
        ("i graduated from MIT in 2020", "identity"),
        ("my favorite color is blue", "identity"),
        ("i speak English and Spanish", "identity"),
        ("my height is 5 feet 10 inches", "identity"),
        ("i am single", "identity"),
        ("i have a cat named Whiskers", "identity"),
        ("my favorite book is Dune", "identity"),
        ("i was born in Chicago", "identity"),
        ("my favorite season is autumn", "identity"),
        ("i am left-handed", "identity"),
        ("my zodiac sign is Leo", "identity"),
        ("i have a brother named Mike", "identity"),
        ("my blood type is O positive", "identity"),
    ]
    
    # ========== EMOTIONAL memories (20 items) ==========
    emotional_data = [
        ("i love pizza", "emotional"),
        ("i hate spiders", "emotional"),
        ("i enjoy playing guitar", "emotional"),
        ("i am afraid of heights", "emotional"),
        ("i like watching movies", "emotional"),
        ("i love the beach", "emotional"),
        ("i hate waking up early", "emotional"),
        ("i enjoy hiking in mountains", "emotional"),
        ("i am scared of the dark", "emotional"),
        ("i like classical music", "emotional"),
        ("i love chocolate cake", "emotional"),
        ("i hate traffic jams", "emotional"),
        ("i enjoy reading science fiction", "emotional"),
        ("i am nervous about public speaking", "emotional"),
        ("i like rainy weather", "emotional"),
        ("i love my family", "emotional"),
        ("i hate being late", "emotional"),
        ("i enjoy solving puzzles", "emotional"),
        ("i am excited about AI", "emotional"),
        ("i like drinking coffee in the morning", "emotional"),
    ]
    
    # ========== SEMANTIC memories (20 items) ==========
    semantic_data = [
        ("python is a programming language", "semantic"),
        ("the capital of France is Paris", "semantic"),
        ("water freezes at 0 degrees Celsius", "semantic"),
        ("the earth orbits the sun", "semantic"),
        ("HTML is used for web pages", "semantic"),
        ("the speed of light is 299 million meters per second", "semantic"),
        ("the great wall of china is in asia", "semantic"),
        ("photosynthesis produces oxygen", "semantic"),
        ("the human heart has 4 chambers", "semantic"),
        ("jupiter is the largest planet", "semantic"),
        ("the chemical symbol for gold is AU", "semantic"),
        ("shakespeare wrote hamlet", "semantic"),
        ("the first moon landing was in 1969", "semantic"),
        ("the pacific ocean is the largest ocean", "semantic"),
        ("the eiffel tower is in paris", "semantic"),
        ("dna stands for deoxyribonucleic acid", "semantic"),
        ("the roman empire fell in 476 AD", "semantic"),
        ("mount everest is the highest mountain", "semantic"),
        ("the square root of 144 is 12", "semantic"),
        ("the great pyramid is in egypt", "semantic"),
    ]
    
    # ========== PROCEDURAL memories (20 items) ==========
    procedural_data = [
        ("to make coffee add hot water to coffee grounds", "procedural"),
        ("to install python go to python.org and download", "procedural"),
        ("to brush teeth use toothpaste and brush for 2 minutes", "procedural"),
        ("to make tea boil water and add tea bag", "procedural"),
        ("to cook rice rinse and boil with 2 parts water", "procedural"),
        ("to tie shoelaces cross and loop", "procedural"),
        ("to send an email open gmail and click compose", "procedural"),
        ("to take a photo open camera app and press shutter", "procedural"),
        ("to bake a cake mix flour eggs and sugar", "procedural"),
        ("to wash hands use soap and water for 20 seconds", "procedural"),
        ("to change a tire loosen bolts and lift the car", "procedural"),
        ("to make an omelet whisk eggs and cook in pan", "procedural"),
        ("to iron clothes heat iron and press fabric", "procedural"),
        ("to plant a seed dig hole and cover with soil", "procedural"),
        ("to change a light bulb turn off power and unscrew", "procedural"),
        ("to wrap a gift fold paper around box and tape", "procedural"),
        ("to make lemonade squeeze lemons and add sugar", "procedural"),
        ("to set an alarm open clock app and choose time", "procedural"),
        ("to sharpen a knife use a sharpening stone", "procedural"),
        ("to make a sandwich put ingredients between bread", "procedural"),
    ]
    
    # ========== EPISODIC memories (20 items) ==========
    episodic_data = [
        ("yesterday i went to the park", "episodic"),
        ("last week i watched a great movie", "episodic"),
        ("on monday i had lunch with my friend", "episodic"),
        ("last month i visited my grandmother", "episodic"),
        ("yesterday i finished reading a book", "episodic"),
        ("last week i started a new project", "episodic"),
        ("on sunday i went swimming", "episodic"),
        ("last year i traveled to Japan", "episodic"),
        ("yesterday i cooked dinner for my family", "episodic"),
        ("last week i learned a new song on guitar", "episodic"),
        ("on friday i went to a concert", "episodic"),
        ("last month i bought a new laptop", "episodic"),
        ("yesterday i met an old friend", "episodic"),
        ("last week i finished a big assignment", "episodic"),
        ("on saturday i went hiking", "episodic"),
        ("last summer i went to the beach", "episodic"),
        ("yesterday i called my parents", "episodic"),
        ("last week i tried a new restaurant", "episodic"),
        ("on thursday i had a job interview", "episodic"),
        ("last month i celebrated my birthday", "episodic"),
    ]
    
    all_data = identity_data + emotional_data + semantic_data + procedural_data + episodic_data
    
    saved_count = 0
    for i, (text, mem_type) in enumerate(all_data, 1):
        print(f"Saving {i:3d}: [{mem_type:10s}] {text[:50]}...")
        memory.add_interaction(text, "")
        saved_count += 1
        if i % 20 == 0:
            print(f"   ... {i}/100 saved")
        time.sleep(0.15)  # Slight delay to avoid overwhelming
    
    print(f"\n✅ Saved {saved_count} memories to LTM\n")
    
    # ==================== PART 2: SEARCH 100 QUERIES ====================
    print("="*70)
    print("🔍 PART 2: SEARCHING 100 QUERIES")
    print("="*70)
    
    # ========== Build search queries based on saved data ==========
    search_tests = []
    
    # Identity queries (20)
    identity_queries = [
        (1, "what is my name?", "John Smith"),
        (2, "how old am i?", "25"),
        (3, "where do i live?", "New York"),
        (4, "what is my job?", "software engineer"),
        (5, "when is my birthday?", "May 15th"),
        (6, "what is my email?", "john@gmail.com"),
        (7, "what is my phone number?", "555-0100"),
        (8, "where did i graduate from?", "MIT"),
        (9, "what is my favorite color?", "blue"),
        (10, "what languages do i speak?", "English and Spanish"),
        (11, "how tall am i?", "5 feet 10 inches"),
        (12, "what is my marital status?", "single"),
        (13, "what pet do i have?", "cat named Whiskers"),
        (14, "what is my favorite book?", "Dune"),
        (15, "where was i born?", "Chicago"),
        (16, "what is my favorite season?", "autumn"),
        (17, "am i left handed or right handed?", "left-handed"),
        (18, "what is my zodiac sign?", "Leo"),
        (19, "do i have any siblings?", "brother named Mike"),
        (20, "what is my blood type?", "O positive"),
    ]
    
    # Emotional queries (20)
    emotional_queries = [
        (21, "what food do i love?", "pizza"),
        (22, "what do i hate?", "spiders"),
        (23, "what instrument do i play?", "guitar"),
        (24, "what am i afraid of?", "heights"),
        (25, "do i like watching movies?", "watching movies"),
        (26, "what place do i love?", "beach"),
        (27, "what do i hate doing?", "waking up early"),
        (28, "what outdoor activity do i enjoy?", "hiking"),
        (29, "what am I scared of?", "dark"),
        (30, "what music do i like?", "classical music"),
        (31, "what dessert do i love?", "chocolate cake"),
        (32, "what do i hate about driving?", "traffic jams"),
        (33, "what genre do i enjoy reading?", "science fiction"),
        (34, "what makes me nervous?", "public speaking"),
        (35, "what weather do i like?", "rainy"),
        (36, "who do i love?", "my family"),
        (37, "what do i hate being?", "late"),
        (38, "what activity do i enjoy solving?", "puzzles"),
        (39, "what am i excited about?", "AI"),
        (40, "what do i like drinking?", "coffee"),
    ]
    
    # Semantic queries (20)
    semantic_queries = [
        (41, "what language is python?", "programming language"),
        (42, "what is the capital of France?", "Paris"),
        (43, "at what temperature does water freeze?", "0 degrees Celsius"),
        (44, "what does the earth orbit?", "sun"),
        (45, "what is HTML used for?", "web pages"),
        (46, "what is the speed of light?", "299 million meters per second"),
        (47, "where is the great wall of china located?", "asia"),
        (48, "what gas does photosynthesis produce?", "oxygen"),
        (49, "how many chambers does the human heart have?", "4 chambers"),
        (50, "what is the largest planet?", "jupiter"),
        (51, "what is the chemical symbol for gold?", "AU"),
        (52, "who wrote hamlet?", "shakespeare"),
        (53, "when was the first moon landing?", "1969"),
        (54, "what is the largest ocean?", "pacific"),
        (55, "where is the eiffel tower located?", "paris"),
        (56, "what does DNA stand for?", "deoxyribonucleic acid"),
        (57, "when did the roman empire fall?", "476 AD"),
        (58, "what is the highest mountain?", "mount everest"),
        (59, "what is the square root of 144?", "12"),
        (60, "where is the great pyramid located?", "egypt"),
    ]
    
    # Procedural queries (20)
    procedural_queries = [
        (61, "how to make coffee?", "add hot water to coffee grounds"),
        (62, "how to install python?", "python.org"),
        (63, "how to brush teeth?", "toothpaste and brush for 2 minutes"),
        (64, "how to make tea?", "boil water"),
        (65, "how to cook rice?", "boil with 2 parts water"),
        (66, "how to tie shoelaces?", "cross and loop"),
        (67, "how to send an email?", "open gmail and click compose"),
        (68, "how to take a photo?", "open camera app and press shutter"),
        (69, "how to bake a cake?", "mix flour eggs and sugar"),
        (70, "how to wash hands?", "soap and water for 20 seconds"),
        (71, "how to change a tire?", "loosen bolts and lift the car"),
        (72, "how to make an omelet?", "whisk eggs and cook in pan"),
        (73, "how to iron clothes?", "heat iron and press fabric"),
        (74, "how to plant a seed?", "dig hole and cover with soil"),
        (75, "how to change a light bulb?", "turn off power and unscrew"),
        (76, "how to wrap a gift?", "fold paper around box and tape"),
        (77, "how to make lemonade?", "squeeze lemons and add sugar"),
        (78, "how to set an alarm?", "open clock app and choose time"),
        (79, "how to sharpen a knife?", "use a sharpening stone"),
        (80, "how to make a sandwich?", "put ingredients between bread"),
    ]
    
    # Episodic queries (20)
    episodic_queries = [
        (81, "what did i do yesterday?", "park"),
        (82, "what happened last week?", "watched a movie"),
        (83, "what did i do on monday?", "had lunch with my friend"),
        (84, "what did i do last month?", "visited my grandmother"),
        (85, "what did i finish yesterday?", "finished reading a book"),
        (86, "what did i start last week?", "started a new project"),
        (87, "what did i do on sunday?", "went swimming"),
        (88, "where did i travel last year?", "Japan"),
        (89, "what did i cook yesterday?", "dinner for my family"),
        (90, "what did i learn last week?", "a new song on guitar"),
        (91, "what did i do on friday?", "went to a concert"),
        (92, "what did i buy last month?", "a new laptop"),
        (93, "who did i meet yesterday?", "an old friend"),
        (94, "what did i finish last week?", "a big assignment"),
        (95, "what did i do on saturday?", "went hiking"),
        (96, "where did i go last summer?", "beach"),
        (97, "who did i call yesterday?", "my parents"),
        (98, "what did i try last week?", "a new restaurant"),
        (99, "what did i do on thursday?", "had a job interview"),
        (100, "what did i celebrate last month?", "my birthday"),
    ]
    
    all_tests = (identity_queries + emotional_queries + semantic_queries + 
                 procedural_queries + episodic_queries)
    
    results = []
    passed = 0
    failed = 0
    
    for test_num, query, expected in all_tests:
        result = memory.search(query, efficient=False)
        actual = result[0] if result else ""
        found = expected.lower() in actual.lower() if actual else False
        
        if found:
            passed += 1
        else:
            failed += 1
        
        results.append((test_num, query, expected, actual, found))
        print_result(test_num, query, expected, actual, found)
        time.sleep(0.1)
    
    # ==================== PART 3: STATISTICS ====================
    print("\n" + "="*70)
    print("📊 PART 3: STATISTICS")
    print("="*70)
    
    # Memory stats
    stats = memory.get_stats()
    print(f"\n📈 Memory Manager Stats:")
    print(f"   STM Size: {stats['stm_size']}/{stats['stm_max']}")
    
    # LTM stats
    total_ltm = sum(idx.ntotal for idx in LTM_index.values())
    print(f"\n💾 LTM Storage:")
    print(f"   Total memories: {total_ltm}")
    for mem_type, idx in LTM_index.items():
        if idx.ntotal > 0:
            print(f"   {mem_type:10s}: {idx.ntotal} memories")
    
    # ==================== PART 4: SAVE TO DISK ====================
    print("\n" + "="*70)
    print("💾 PART 4: SAVING TO DISK")
    print("="*70)
    memory.save_all()
    print("✅ Memory saved to disk")
    
    # ==================== FINAL RESULTS ====================
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    
    success_rate = (passed / 100) * 100
    print(f"\n🏆 Test Results: {passed}/100 passed ({success_rate:.1f}%)")
    
    # Results by category
    print("\n📋 Results by category:")
    
    def calc_category_rate(queries, tests, results):
        cat_passed = 0
        for i in range(len(queries)):
            test_num = queries[i][0]
            for r in results:
                if r[0] == test_num and r[4]:
                    cat_passed += 1
                    break
        return cat_passed
    
    identity_passed = calc_category_rate(identity_queries, identity_data, results)
    emotional_passed = calc_category_rate(emotional_queries, emotional_data, results)
    semantic_passed = calc_category_rate(semantic_queries, semantic_data, results)
    procedural_passed = calc_category_rate(procedural_queries, procedural_data, results)
    episodic_passed = calc_category_rate(episodic_queries, episodic_data, results)
    
    print(f"   🆔 Identity   : {identity_passed}/20 ({identity_passed*5:.0f}%)")
    print(f"   ❤️ Emotional  : {emotional_passed}/20 ({emotional_passed*5:.0f}%)")
    print(f"   📚 Semantic   : {semantic_passed}/20 ({semantic_passed*5:.0f}%)")
    print(f"   ⚙️ Procedural : {procedural_passed}/20 ({procedural_passed*5:.0f}%)")
    print(f"   📅 Episodic   : {episodic_passed}/20 ({episodic_passed*5:.0f}%)")
    
    # Show failed tests
    failed_tests = [r for r in results if not r[4]]
    if failed_tests:
        print(f"\n❌ Failed tests ({len(failed_tests)}):")
        for r in failed_tests[:10]:  # Show first 10 failures
            print(f"   Test {r[0]:3d}: {r[1][:40]} → expected: {r[2]}")
        if len(failed_tests) > 10:
            print(f"   ... and {len(failed_tests) - 10} more failures")
    
    # Final verdict
    print("\n" + "="*70)
    if success_rate == 100:
        print("🎉 PERFECT! All 100 tests passed!")
        print("🧠 Memory system is perfectly working!")
    elif success_rate >= 90:
        print(f"🎉 EXCELLENT! {success_rate:.0f}% passed. Memory system is very reliable!")
    elif success_rate >= 80:
        print(f"👍 GOOD! {success_rate:.0f}% passed. Memory system works well!")
    elif success_rate >= 70:
        print(f"⚠️ FAIR! {success_rate:.0f}% passed. Needs minor improvements.")
    else:
        print(f"❌ POOR! {success_rate:.0f}% passed. Needs major improvements.")
    
    print("="*70)
    print("✅ TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()