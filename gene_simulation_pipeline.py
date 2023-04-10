import sys
from optparse import OptionParser
import numpy as np
import scipy.stats
import gzip
import subprocess
import math

usage = "usage: %prog [options]"
parser = OptionParser(usage)
parser.add_option("", "--bim-file", default=None)  # all SNPs in the region
parser.add_option("", "--ld-file", default=None)  # in row form
parser.add_option("", "--ld-threshold", default=0, type="float")
parser.add_option("", "--h2", default=0.30, type="float")  # 1-h2 is added as phenotypic variance
parser.add_option("", "--N", default=100000, type="float")  # sample size for sumstat simulations
parser.add_option("", "--total_snp_genome", default=9997231, type="int")  # total number of snps in genome
parser.add_option("", "--beta-file", default=None)  # specify mapping from SNP to true beta
parser.add_option("", "--beta-dist-file", default=None)  # specify a distribution of beta values to be assigned to
# each SNP within a region. No header, columns are chrom, begin, end, p, mean, var [replicate] where if the optional
# [1-based] replicate column is present, it will apply the region only to the specified replicate. If multiple lines
# apply to a SNP, it will take the first one
parser.add_option("", "--max-component-size", type="int",
                  default=np.inf)  # control the maximum number of SNPs to be included in an LD-block (smaller is
# faster)
parser.add_option("", "--gene-loc", default=None) # File that contains the gene location information
parser.add_option("", "--assume-r", action="store_true")
parser.add_option("", "--num-sim", type="int", default=1)  # generate statistics for multiple independent replicates
parser.add_option("", "--ldsc-format", action="store_true")  # output sumstats in LDSC format
parser.add_option("", "--num-causal-snps-out",
                  default=None)  # write a file with the number of causal SNPs per iteration
parser.add_option("", "--output-file", default=None)
parser.add_option("", "--file-path", default=None)  # specify the path of the file for outputs
parser.add_option("", "--p", type="float", default=1.0)  # Proportion of values going into the random distribution
parser.add_option("", "--p-causal-gene", type="float", default=0.4)  # Proportion of values going into the random distribution
parser.add_option("", "--p-outside-causal-gene", type="float", default=0.0001)  # Proportion of values going into the random distribution
parser.add_option("", "--chrom")  # Tells use the chromosome number that we're examining
(options, args) = parser.parse_args()


def bail(message):
    sys.stderr.write("%s\n" % message)
    sys.exit(1)


def warn(message):
    sys.stderr.write("Warning: %s\n" % message)


def log(message):
    sys.stderr.write("%s\n" % message)


def open_gz(file, sort_col=None, reverse=True, header=False):
    if sort_col is None:
        if file[-3:] == ".gz":
            fh = gzip.open(file)
            fh.readline()
        else:
            fh = open(file)
            fh.readline()
    else:
        reverse_flag = ""
        if reverse:
            reverse_flag = "r"

        if file[-3:] == ".gz":
            return subprocess.Popen(
                ["zcat %s | tail -n+%d | sort -g%sk%s" % (file, 1 if header else 2, reverse_flag, sort_col)],
                stdout=subprocess.PIPE, shell=True).stdout
        else:
            return subprocess.Popen(
                ["tail -n+%d %s | sort -g%sk%s" % (1 if header else 2, file, reverse_flag, sort_col)],
                stdout=subprocess.PIPE, shell=True).stdout


# Used to determine the causal variants by finding the maximum true beta that a causal variant can have
# if we want to see all snps just make the maximum value = 0 whereas the other one would have the minimum
# at the location of the causal snp
def calculateMinBetaValue(true_beta_all_snps, causal_snps, allSnps):
    absolute_value_all_snps = np.absolute(true_beta_all_snps)
    ranked_all_snps = np.sort(absolute_value_all_snps)
    ranked_all_snps = np.flip(ranked_all_snps)
    minimum_value = ranked_all_snps[causal_snps - 1]
    if allSnps:
        return 0
    else:
        return minimum_value


# variables  -----------------------------------------------------------------------------------------------------------
assume_r = options.assume_r
ld_file = options.ld_file
# ----------------------------------------------------------------------------------------------------------------------

if ld_file is None:
    bail("Need --ld-file")

if options.h2 < 0 or options.h2 > 1:
    bail("--h2 must be between 0 and 1")

print("Reading ld file... ", ld_file)
ld_rev_fh = open_gz(ld_file, 7, True)
ld_for_fh = open_gz(ld_file, 7, False)
snp_to_component = {}
component_to_snp = {}
index = 0
component = 0
max_component_size = 100
chr_to_snp_pos = {}
snp_to_chr_pos = {}
snp_to_alt = {}
snp_to_ref = {}
chr_pos_to_snp = {}

line_rev = None
valid_rev = True
line_for = None
valid_for = True
count = 0

# Used to calculate the variance later
total_number_snps = 0
snp_container = []

while True:
    count += 1
    # read through the file forwards and backwards at the same time, taking the highest abs
    if valid_rev and line_rev is None:
        line_rev = ld_rev_fh.readline()
        cols_rev = line_rev.strip().split()
        if len(cols_rev) != 7:
            log("Bad line number " + str(count))
            break
        value_rev = float(cols_rev[6])
        if value_rev > 0:
            valid_rev = True
        else:
            valid_rev = False
    if valid_for and line_for is None:
        line_for = ld_for_fh.readline()
        cols_for = line_for.strip().split()
        if len(cols_for) != 7:
            log("Bad line number " + str(count))
            break
        value_for = float(cols_for[6])
        if value_for < 0:
            valid_for = True
        else:
            valid_for = False

    if valid_rev and (not valid_for or abs(value_rev) >= abs(value_for)):
        line = line_rev
        cols = cols_rev
        value = value_rev
        line_rev = None
    elif valid_for and (not valid_rev or abs(value_rev) < abs(value_for)):
        line = line_for
        cols = cols_for
        value = value_for
        line_for = None
    else:
        break

    if value < 0:
        assume_r = True

    if abs(value) < options.ld_threshold:  # if is not, I think it will also continue
        continue
    snp_1 = cols[2]
    snp_2 = cols[5]
    if type(snp_1) == bytes:
        snp_1 = bytes.decode(snp_1)
    if type(snp_2) == bytes:
        snp_2 = bytes.decode(snp_2)

    snp_1_chr = bytes.decode(cols[0])
    snp_1_pos = int(cols[1])
    snp_to_chr_pos[snp_1] = (snp_1_chr, snp_1_pos)
    snp_2_chr = bytes.decode(cols[3])
    snp_2_pos = int(cols[4])
    snp_to_chr_pos[snp_2] = (snp_2_chr, snp_2_pos)

    if snp_1_chr not in chr_to_snp_pos:
        chr_to_snp_pos[snp_1_chr] = set()
    chr_to_snp_pos[snp_1_chr].add(snp_1_pos)
    chr_pos_to_snp[(snp_1_chr, snp_1_pos)] = snp_1
    if snp_2_chr not in chr_to_snp_pos:
        chr_to_snp_pos[snp_2_chr] = set()
    chr_to_snp_pos[snp_2_chr].add(snp_2_pos)
    chr_pos_to_snp[(snp_2_chr, snp_2_pos)] = snp_2

    if snp_1 not in snp_to_component and snp_2 not in snp_to_component:
        component += 1
        snp_to_component[snp_1] = component
        snp_to_component[snp_2] = component
        component_to_snp[component] = set()
        component_to_snp[component].add(snp_1)
        component_to_snp[component].add(snp_2)
    elif snp_1 in snp_to_component and snp_2 not in snp_to_component:
        if len(component_to_snp[snp_to_component[snp_1]]) < options.max_component_size:
            snp_to_component[snp_2] = snp_to_component[snp_1]
            component_to_snp[snp_to_component[snp_1]].add(snp_2)
        else:
            component += 1
            snp_to_component[snp_2] = component
            component_to_snp[component] = set()
            component_to_snp[component].add(snp_2)
    elif snp_2 in snp_to_component and snp_1 not in snp_to_component:
        if len(component_to_snp[snp_to_component[snp_2]]) < options.max_component_size:
            snp_to_component[snp_1] = snp_to_component[snp_2]
            component_to_snp[snp_to_component[snp_2]].add(snp_1)
        else:
            component += 1
            snp_to_component[snp_1] = component
            component_to_snp[component] = set()
            component_to_snp[component].add(snp_1)
    elif snp_2 in snp_to_component and snp_1 in snp_to_component and snp_to_component[snp_1] != snp_to_component[snp_2]:
        if len(component_to_snp[snp_to_component[snp_1]]) + len(
                component_to_snp[snp_to_component[snp_2]]) <= options.max_component_size:
            component_1 = snp_to_component[snp_1]
            component_2 = snp_to_component[snp_2]
            for snp in component_to_snp[component_2]:
                snp_to_component[snp] = component_1
            component_to_snp[component_1] = component_to_snp[component_1].union(component_to_snp[component_2])
            component_to_snp.pop(component_2)

    if len(component_to_snp[snp_to_component[snp_1]]) > max_component_size:
        max_component_size = len(component_to_snp[snp_to_component[snp_1]])
log("Max component size: %s" % max_component_size)

ld_for_fh.close()
ld_rev_fh.close()

if options.bim_file:
    bim_fh = open(options.bim_file)
    for line in bim_fh:
        total_number_snps = total_number_snps + 1
        cols = line.strip().split()
        snp = cols[1]
        chr = cols[0]
        pos = int(cols[3])

        snp_to_alt[snp] = str(cols[4])
        snp_to_ref[snp] = str(cols[5])

        snp_container.append(snp)

        if snp not in snp_to_chr_pos:
            snp_to_chr_pos[snp] = (chr, pos)
            if chr not in chr_to_snp_pos:
                chr_to_snp_pos[chr] = set()
            chr_to_snp_pos[chr].add(pos)
            if (chr, pos) not in chr_pos_to_snp:
                chr_pos_to_snp[(chr, pos)] = snp

            if snp not in snp_to_component:
                component += 1
                snp_to_component[snp] = component
                component_to_snp[component] = set()
                component_to_snp[component].add(snp)

    bim_fh.close()

for chrom in chr_to_snp_pos:
    chr_to_snp_pos[chrom] = sorted(list(chr_to_snp_pos[chrom]))

snp_to_index = {}
component_to_cor = {}
for component in component_to_snp:
    index = 0
    # print "%s %s" % (component, component_to_snp[component])
    for snp in sorted(component_to_snp[component]):
        snp_to_index[snp] = index
        index += 1
    # print("%s %s" % (component, len(component_to_snp[component])))
    component_to_cor[component] = np.identity(len(component_to_snp[component]), dtype=np.float64)

ld_fh = open_gz(ld_file, 7)
for line in ld_fh:
    cols = line.strip().split()
    value = float(cols[6])
    if abs(value) < options.ld_threshold:
        continue
    if not assume_r:
        value = math.sqrt(value)
    snp_1 = cols[2]
    snp_2 = cols[5]
    if type(snp_1) == bytes:
        snp_1 = bytes.decode(snp_1)
    if type(snp_2) == bytes:
        snp_2 = bytes.decode(snp_2)

    if snp_to_component[snp_1] == snp_to_component[snp_2]:
        component_to_cor[snp_to_component[snp_1]][snp_to_index[snp_1], snp_to_index[snp_2]] = value
        component_to_cor[snp_to_component[snp_1]][snp_to_index[snp_2], snp_to_index[snp_1]] = value
ld_fh.close()

snp_to_true_beta_params = {}

# Generate the true betas based on variance and proportion fraction
if options.beta_dist_file:

    import random
    import bisect

    beta_dist_fh = open(options.beta_dist_file)
    for line in beta_dist_fh:
        cols = line.strip().split()
        if len(cols) != 6 and len(cols) != 7:
            warn(
                "Ignoring line without six columns for (chrom, start, end, p, mean, var [replicate]):  %s" % line.strip())
            continue
        chrom = cols[0]
        start = int(cols[1])
        end = int(cols[2])
        p = float(cols[3])
        if p < 0 or p > 1:
            log("Error: bad value for bernoulli (%s)" % (p))
            continue
        mean = float(cols[4])
        var = float(cols[5])
        if var < 0:
            log("Error: bad value for var (%s)" % (var))
            continue

        if len(cols) == 7:
            rep = int(cols[6])
            if rep < 1:
                log("Error: bad value for rep (%s)" % (rep))
                continue
        else:
            rep = 1

        # find all overlapping snps

        if chrom in chr_to_snp_pos:
            start_ind = bisect.bisect_left(chr_to_snp_pos[chrom], start)
            for ind in range(start_ind, len(chr_to_snp_pos[chrom])):
                if chr_to_snp_pos[chrom][ind] < start:
                    log("There is a bug -- %s is less than %s" % (chr_to_snp_pos[chrom][ind], start))
                    continue
                if chr_to_snp_pos[chrom][ind] > end:
                    break
                cur_snp = chr_pos_to_snp[(chrom, chr_to_snp_pos[chrom][ind])]
                if cur_snp not in snp_to_true_beta_params:
                    snp_to_true_beta_params[cur_snp] = []
                snp_to_true_beta_params[cur_snp].append((p, mean, var, rep - 1))

snp_to_true_beta = {}
if options.gene_loc:
    gene_fh = open(options.gene_loc)

    for line in gene_fh:
        cols = line.strip().split()
        chromosome = cols[1]
        if chromosome == options.chrom:
            print("Currently", line)
    gene_fh.close()

if options.beta_file:
    beta_fh = open(options.beta_file)
    for line in beta_fh:
        cols = line.strip().split()
        if len(cols) != 2:
            warn("Ignoring line without two columns for (snp, beta):  %s" % line.strip())
            continue
        snp = cols[0]
        if snp in snp_to_index:
            true_beta = float(cols[1])
            snp_to_true_beta[snp] = true_beta
    beta_fh.close()

snp_to_beta = {}
snp_to_p = {}
locus_fh = open(str(options.output_file) + ".locus", 'w')
gwas_fh = open(str(options.output_file) + ".gwas", 'w')

if options.ldsc_format:
    sys.stdout.write("SNP\tReplicate\tA1\tA2\tZ\tN\n")
else:
    header_locus = "SNP\tChrom\tPos\tRef\tAlt\tReplicate\tEffect\tStdErr\tP-value\n"
    header_gwas = "MarkerName\tChrom\tPos\tRef\tAlt\tWeight\tGC.Zscore\tGC.Pvalue\tOverall\tDirection\tEffect\tStdErr\n"
    locus_fh.write(header_locus)
    gwas_fh.write(header_gwas)

num_true_causal_snps = {}
snp_true_beta_container = {}

# start of the simulation by generating true betas and calculating marginals
if True:

    import random

    # the reason not var = h2 / number of snps in gene because trying to make heritability per snp so evenly distributed
    p = options.p
    mean = 0
    h2_proportion = options.h2 / (options.total_snp_genome * p)
    var = h2_proportion
    causal_snps = math.ceil(total_number_snps * p)

    # List of all the snps
    random.shuffle(snp_container)
    snp_container_causal = snp_container[0:int(causal_snps-1)]

    for component in component_to_cor:

        cor_matrix = component_to_cor[component]

        # Associate each snp with a true beta
        for it in range(0, options.num_sim):

            if it not in num_true_causal_snps:
                num_true_causal_snps[it] = 0

            cur_true_beta = np.zeros(len(component_to_snp[component]))
            cur_geno_var = np.ones(len(component_to_snp[component])) * (1 - options.h2)

            # If data was obtained from beta_dist_file
            if len(snp_to_true_beta_params) > 0:
                for snp in component_to_snp[component]:
                    if snp in snp_to_true_beta_params:
                        for true_beta_params in snp_to_true_beta_params[snp]:
                            if true_beta_params[3] is None or true_beta_params[3] == it:
                                if random.random() <= true_beta_params[0]:
                                    cur_true_beta[snp_to_index[snp]] = \
                                        np.random.normal(loc=true_beta_params[1], scale=np.sqrt(true_beta_params[2]),
                                                         size=1)[0]
                                    snp_true_beta_container[snp] = cur_true_beta[snp_to_index[snp]]
                                break

            # If no beta_dist_file then use the mean,variance from h2 provided and sample size
            else:
                for snp in component_to_snp[component]:
                    if snp in snp_container_causal:
                        cur_true_beta[snp_to_index[snp]] = np.random.normal(mean, scale=np.sqrt(var), size=1)[0]
                        snp_true_beta_container[snp] = cur_true_beta[snp_to_index[snp]]
                    else:
                        snp_true_beta_container[snp] = 0

            # if the true beta was provided in a different file
            if len(snp_to_true_beta) > 0:
                for snp in component_to_snp[component]:
                    if snp in snp_to_true_beta:
                        cur_true_beta[snp_to_index[snp]] = snp_to_true_beta[snp]
                        snp_true_beta_container[snp] = cur_true_beta[snp_to_index[snp]]

            # keeps track of all the causal snps per simulation
            num_true_causal_snps[it] += np.sum(cur_true_beta != 0)

            # noise term
            v = np.random.normal(loc=0, scale=1, size=len(component_to_snp[component]))
            # v = np.ones(len(component_to_snp[component])) * 0.12

            # Start calculation for marginal betas to compare with true betas
            M = cor_matrix.shape[0]
            sigma_e = 1 - options.h2

            beta_corr = np.diag(np.ones(M) * sigma_e / options.N)
            L2 = np.linalg.cholesky(beta_corr)

            marginal_beta_method_2 = (np.matmul(cor_matrix, cur_true_beta)) + np.matmul(L2, v)
            standard_error_method_2_alt = np.sqrt(cur_geno_var / options.N)
            Z_score_method_2 = marginal_beta_method_2 / standard_error_method_2_alt

            for snp in component_to_snp[component]:
                if len(component_to_snp[component]) == 1:
                    beta_method_2_ind = marginal_beta_method_2[0]
                else:
                    beta_method_2_ind = marginal_beta_method_2[snp_to_index[snp]]

                se_method_2_ind = standard_error_method_2_alt[snp_to_index[snp]]

                z_score_method_2_ind = Z_score_method_2[snp_to_index[snp]]

                pvalue_method_2 = str(2 * scipy.stats.norm.sf(abs(z_score_method_2_ind)))

                if options.ldsc_format:
                    sys.stdout.write("%s\t%d\t%s\t%s\t%.3g\t%d\n" % (
                        snp, it + 1, "R", "A", beta_method_2_ind / se_method_2_ind, options.N))
                else:
                    chrom = str(snp_to_chr_pos[snp][0])
                    pos = str(snp_to_chr_pos[snp][1])
                    alt = snp_to_alt[snp]
                    ref = snp_to_ref[snp]
                    N = str(options.N)

                    if snp in snp_true_beta_container:
                        temp_true_beta_holder = snp_true_beta_container[snp]
                        snp_true_beta_container[snp] = [snp, chrom, temp_true_beta_holder, pos, ref, alt]

                    line_gwas = snp + "\t" + chrom + "\t" + pos + "\t" + ref + "\t" + alt + "\t" + N + "\t" + str(
                        z_score_method_2_ind) + "\t" + pvalue_method_2 + "\t" + "+\t+-+-+\t" + str(
                        beta_method_2_ind) + "\t" + str(se_method_2_ind)
                    line_locus = snp + "\t" + chrom + "\t" + pos + "\t" + ref + "\t" + alt + "\t" + str(
                        it + 1) + "\t" + str(beta_method_2_ind) + "\t" + str(se_method_2_ind) + "\t" + pvalue_method_2
                    gwas_fh.write(line_gwas + "\n")
                    locus_fh.write(line_locus + "\n")

    # Store all the snps and associated betas into a file. Sort them by the absolute value of the betas
    file_output = open(str(options.output_file) + ".true_beta", "w")
    file_output.write("SNP\tChrom\tTrue Beta\tPos\tRef\tAlt\n")
    for key, value in sorted(snp_true_beta_container.items(), key=lambda k: abs(k[1][2]), reverse=True):
        file_output.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(value[0], value[1], value[2], value[3], value[4], value[5]))
    file_output.close()


if options.num_causal_snps_out is not None:
    out_suma_fh = open(options.num_causal_snps_out, 'w')
    out_suma_fh.write("Replicate\tNum_Causal\n")
    for it in range(0, options.num_sim):
        out_suma_fh.write("%d\t%d\n" % (it + 1, num_true_causal_snps[it] if it in num_true_causal_snps else 0))
    out_suma_fh.close()

print('Number of SNPs simulated: ', np.sum([len(component_to_snp[comp]) for comp in component_to_cor]))
locus_fh.close()
gwas_fh.close()
exit()
