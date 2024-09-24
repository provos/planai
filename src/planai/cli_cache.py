import sys

from diskcache import Cache


def handle_cache_subcommand(parsed_args):
    cache = Cache(parsed_args.cache_dir)
    if parsed_args.search_dirs:
        for search_dir in parsed_args.search_dirs:
            sys.path.append(search_dir)

    if parsed_args.clear:
        cache.clear()
        print("Cache cleared successfully.")
    elif parsed_args.delete:
        to_delete = []
        for key in cache:
            try:
                value, input_data = cache.get(key)
            except Exception as e:
                print(f"Error handling cache: {str(e)}", file=sys.stderr)
                continue

            if input_data.name == parsed_args.delete:
                to_delete.append(key)
        print(f"Deleting {len(to_delete)} entries.")
        for key in to_delete:
            del cache[key]
    else:
        total_size = 0
        print(f"Reading cache from: {parsed_args.cache_dir}")
        print("-----------------------------")

        if len(cache) == 0:
            print("The cache is empty.")
        else:
            # Dictionary to hold the statistics
            stats = {}

            for key in cache:
                try:
                    value, input_data = cache.get(key)
                except Exception as e:
                    print(f"Error handling cache: {str(e)}", file=sys.stderr)
                    continue

                input_name = input_data.name

                if parsed_args.output_task_filter:
                    if not any(
                        [task.name == parsed_args.output_task_filter for task in value]
                    ):
                        continue

                # Update statistics
                if input_name not in stats:
                    stats[input_name] = {"count": 0, "total_size": 0}

                stats[input_name]["count"] += 1
                # Estimate size of cached entry
                stats[input_name]["total_size"] += sum(sys.getsizeof(v) for v in value)

            # Print the statistics
            for input_name, data in stats.items():
                print(f"Input: {input_name}")
                print(f"  Number of Entries: {data['count']}")
                print(f"  Total Size (bytes): {data['total_size']}")
                print("-----------------------------")

                total_size += data["total_size"]
            print(f"Total Size of Cache (bytes): {total_size}")
