# PathMNIST dataset audit

## Partition: iid
- official split sizes: {'train': 89996, 'val': 10004, 'test': 7180}
- server_ref: 8999 samples, class_counts=[928, 941, 1099, 1067, 810, 1159, 778, 913, 1304]
- calibration (=val, untouched): 10004 samples, class_counts=[1041, 1057, 1152, 1156, 890, 1354, 877, 1045, 1432]
- test (untouched): 7180 samples, class_counts=[1338, 847, 339, 634, 1035, 592, 741, 421, 1233]
- per-client:
  - client 0: n=16200, class_counts=[1641, 1649, 1861, 1864, 1443, 2305, 1456, 1708, 2273]
  - client 1: n=16200, class_counts=[1630, 1765, 1899, 1925, 1397, 2188, 1374, 1700, 2322]
  - client 2: n=16199, class_counts=[1745, 1806, 1814, 1800, 1438, 2158, 1410, 1681, 2347]
  - client 3: n=16199, class_counts=[1745, 1700, 1832, 1847, 1412, 2229, 1440, 1702, 2292]
  - client 4: n=16199, class_counts=[1677, 1648, 1855, 1898, 1506, 2143, 1428, 1697, 2347]
- disjointness proof (all overlap counts must be 0): {'server_ref_vs_clients': 0, 'client_vs_client': 0, 'content_hash_train_vs_val': 0, 'content_hash_train_vs_test': 0, 'content_hash_val_vs_test': 0, 'content_hash_server_ref_vs_client_training': 0, 'content_hash_server_ref_vs_val': 0, 'content_hash_server_ref_vs_test': 0, 'content_hash_client_training_vs_val': 0, 'content_hash_client_training_vs_test': 0, 'all_zero': True}

## Partition: dirichlet (alpha=0.5)
- official split sizes: {'train': 89996, 'val': 10004, 'test': 7180}
- server_ref: 8999 samples, class_counts=[928, 941, 1099, 1067, 810, 1159, 778, 913, 1304]
- calibration (=val, untouched): 10004 samples, class_counts=[1041, 1057, 1152, 1156, 890, 1354, 877, 1045, 1432]
- test (untouched): 7180 samples, class_counts=[1338, 847, 339, 634, 1035, 592, 741, 421, 1233]
- per-client:
  - client 0: n=8071, class_counts=[2368, 119, 4552, 0, 0, 605, 81, 192, 154]
  - client 1: n=13146, class_counts=[3769, 73, 74, 2113, 12, 3602, 8, 94, 3401]
  - client 2: n=15222, class_counts=[26, 494, 12, 14, 425, 508, 5077, 7555, 1111]
  - client 3: n=29413, class_counts=[2225, 2436, 3655, 3, 6544, 6210, 1067, 382, 6891]
  - client 4: n=15145, class_counts=[50, 5446, 968, 7204, 215, 98, 875, 265, 24]
- disjointness proof (all overlap counts must be 0): {'server_ref_vs_clients': 0, 'client_vs_client': 0, 'content_hash_train_vs_val': 0, 'content_hash_train_vs_test': 0, 'content_hash_val_vs_test': 0, 'content_hash_server_ref_vs_client_training': 0, 'content_hash_server_ref_vs_val': 0, 'content_hash_server_ref_vs_test': 0, 'content_hash_client_training_vs_val': 0, 'content_hash_client_training_vs_test': 0, 'all_zero': True}
