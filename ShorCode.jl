__precompile__()

module ShorCode

export CountCoincidenceShorCode

push!(LOAD_PATH, ".");
using quantum

hadamard = [1 1; 1 -1]/sqrt(2.);

ShorStabilizerGroupeOp = Array[ sigma_z, sigma_z, sigma_z, sigma_z,
                            sigma_z, sigma_z, sigma_x, sigma_x]
ShorStabilizerGroupeIdx = Array[[1 0],[2 1],[4 3],[5 4],[7 6],[8 7],[5 4 3 2 1 0],[8 7 6 5 4 3]]

ShorBitFlipStabilizerCorrectionEig = Array[
    [-1; 1; 1; 1; 1; 1],
    [-1; -1; 1; 1; 1; 1],
    [1; -1; 1; 1; 1; 1],
    [1; 1; -1; 1; 1; 1],
    [1; 1; -1; -1; 1; 1],
    [1; 1; 1; -1; 1; 1],
    [1; 1; 1; 1; -1; 1],
    [1; 1; 1; 1; -1; -1],
    [1; 1; 1; 1; 1; -1]
]
ShorBitFlipStabilizerCorrectionIdx = [0;1;2;3;4;5;6;7;8]

ShorPhaseFlipStabilizerCorrectionEig = Array[[-1;1], [-1;-1], [1;-1]]
ShorPhaseFlipStabilizerCorrectionIdx = Array[[0 1 2], [3 4 5], [6 7 8]];

function concat_ancilla(psi::Array{Complex{Float64},2}, n::Int64)
    m = log2(length(psi))
    psi_anc = zeros(Complex{Float64}, Int(2^(n+m)),1)
    psi_anc[1:length(psi)] = psi
    return psi_anc
end

function apply_control_not!(psi::Array{Complex{Float64},2}, control_qubit::Int64, target_qubit::Int64)
    psi_i_temp = copy(psi)
    mov = max(control_qubit,target_qubit)
    for i = 0:length(psi)-1
        if testbit(i,control_qubit) & !testbit(i,target_qubit)
            psi[i+1] = psi_i_temp[i+1+2^mov]
        elseif testbit(i,control_qubit) & testbit(i,target_qubit)
            psi[i+1] = psi_i_temp[i+1-2^mov]
        end
    end
end

function apply_toffoli!(psi::Array{Complex{Float64},2}, control_qubit1::Int64, control_qubit2::Int64, target_qubit::Int64)
    psi_i_temp = copy(psi)
    mov = target_qubit
    for i = 0:length(psi)-1
        if testbit(i,control_qubit1) & testbit(i,control_qubit2) & !testbit(i,target_qubit)
            psi[i+1] = psi_i_temp[i+1+2^mov]
        elseif testbit(i,control_qubit1) & testbit(i,control_qubit2) & testbit(i,target_qubit)
            psi[i+1] = psi_i_temp[i+1-2^mov]
        end
    end
end

function Shor_encoder(psi::Array{Complex{Float64},2})
    psi_encode = concat_ancilla(psi, 8)
    apply_control_not!(psi_encode, 0, 3)
    apply_control_not!(psi_encode, 0, 6)

    for i in 0:2
        apply_unitary!(psi_encode, hadamard, 3*i)
    end
    for i in 0:2
        apply_control_not!(psi_encode, 3*i, 3*i+1)
    end
    for i in 0:2
        apply_control_not!(psi_encode, 3*i, 3*i+2)
    end
    return psi_encode
end

function ShorCodeQEC(psi::Array{Complex{Float64},2})
    Eigenvals = zeros(Int64,length(ShorStabilizerGroupeIdx))

    for i in 1:length(ShorStabilizerGroupeIdx)
        psi_test = copy(psi)
        for j in 1:length(ShorStabilizerGroupeIdx[i])
            apply_unitary!(psi_test, ShorStabilizerGroupeOp[i], ShorStabilizerGroupeIdx[i][j])
        end
        if psi_test == psi
            Eigenvals[i] = 1
        elseif psi_test == -psi
            Eigenvals[i] = -1
        end
    end

    psi_corr = copy(psi)
    for i in 1:length(ShorBitFlipStabilizerCorrectionEig)
        if Eigenvals[1:6] == ShorBitFlipStabilizerCorrectionEig[i]
            apply_unitary!(psi_corr, sigma_x, ShorBitFlipStabilizerCorrectionIdx[i])
        end
    end
    for i in 1:length(ShorPhaseFlipStabilizerCorrectionEig)
        if Eigenvals[7:8] == ShorPhaseFlipStabilizerCorrectionEig[i]
            for j in 1:3
                apply_unitary!(psi_corr, sigma_z, ShorPhaseFlipStabilizerCorrectionIdx[i][j])
            end
        end
    end

    for i in 0:2
        apply_control_not!(psi_corr, 3*i, 3*i+1)
    end

    for i in 0:2
        apply_control_not!(psi_corr, 3*i, 3*i+2)
    end

    for i in 0:2
        apply_toffoli!(psi_corr, 3*i+2, 3*i+1, 3*i)
    end

    for i in 0:2
        apply_unitary!(psi_corr, hadamard, 3*i)
    end

    apply_control_not!(psi_corr, 0, 6)
    apply_control_not!(psi_corr, 0, 3)
    apply_toffoli!(psi_corr, 6, 3, 0)

    return psi_corr
end

function CountCoincidenceShorCode(p::Array{Float64,1}, case::AbstractString, N=10000)
    coincidences = SharedArray(Float64, length(p), init=0)
    n = 1
    for j in 1:length(p)
        for i=1:N
            psi = random_state(2)
            if case == "QEC"
                psi_enc = Shor_encoder(psi)
                psi_enc_err1 = Error_generator(sigma_x, psi_enc, p[j])
                psi_enc_err2 = Error_generator(sigma_z, psi_enc_err1, p[j])

                psi_enc_corrected = ShorCodeQEC(psi_enc_err2)

                if norm(psi - psi_enc_corrected[1:2]) < 1e-6
                    coincidences[j] += 1.
                elseif norm(psi - psi_enc_corrected[1:2] - 2) < 1e-6
                    coincidences[j] += 1.
                end
            elseif case == "NoQEC"
                psi_err = Error_generator(sigma_x, psi, p[j])
                psi_enc_err = Error_generator(sigma_z, psi_err, p[j])
                if psi == psi_enc_err
                    coincidences[j] += 1.
                end
            end
        end
    end
    return coincidences/N
end


end
