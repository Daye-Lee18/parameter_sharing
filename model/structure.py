import math 

N = 12
M = 3 

# mode = "cycle_rev"
# mode = "sequence"
mode = "cycle"

##### start from 0 index 
cnt = 0
for i in range(N):
    if i == 0: 
        print(f"new: {cnt}")
    elif mode == "sequence":
        if (i) % math.floor(N/M) == 0:
            cnt += 1 
            print(f"new: {cnt}")
        else:
            print(f"shared: {cnt}")
    elif mode == "cycle":
        if i < M: 
            cnt += 1
            print(f"new: {cnt}")
        else:
            print(f"shared: { ((i) % M )}")
    elif mode == "cycle_rev":
        if i < M:
            cnt += 1 
            print(f"new: {cnt}")
        elif (i) < M * (round(N/M,0)-1):
            print(f"shared_a: {((i)%M)}")
        else:
            print(f"shared_b: {M - ((i)%M) -1}")


################# index start with 1 
# cnt = 0
# for i in range(1, N+1):
#     if i == 1: 
#         cnt += 1 
#         print(f"new: {cnt}")
#     elif mode == "sequence":
#         if (i-1) % math.floor(N/M) == 0:
#             cnt += 1 
#             print(f"new: {cnt}")
#         else:
#             print(f"shared: {cnt}")
#     elif mode == "cycle":
#         if i <= M:
#             cnt += 1 
#             print(f"new: {cnt}")
#         else:
#             print(f"shared: { ((i-1) % M ) +1}")
#     elif mode == "cycle_rev":
#         if i <= M:
#             cnt += 1 
#             print(f"new: {cnt}")
#         elif i <= M * (round(N/M,0) - 1):
#             print(f"shared_a: {((i-1)%M) + 1}")
#         else:
#             print(f"shared_b: {M - ((i-1)%M)}")
