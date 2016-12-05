__precompile__()
push!(LOAD_PATH, ".");

module ShorCode
export run_code

using quantum
using PyPlot

hadamard = [1 1; 1 -1]/sqrt(2)
function concat_ancilla(psi::Array{Complex{Float64},2}, n::Int64)
    psi_anc = zeros(Complex{Float64}, 2^(n+1))
    psi_anc[1:length(psi)] = psi
    return psi_anc
end

function apply_control_not!(psi::Array{Complex{Float64},1}, control_qubit::Int64, target_qubit::Int64)
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

function apply_toffoli!(psi::Array{Complex{Float64},1}, control_qubit1::Int64, control_qubit2::Int64, target_qubit::Int64)
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

function encoder(psi::Array{Complex{Float64},2})
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

function error_generator(Err, psi::Array{Complex{Float64},1}, p::Float64)
    n = Int(log2(length(psi)))
    psi_err = copy(psi)
    for i=0:n-1
        if rand()< p
            err = rand(Err)
            apply_unitary!(psi_err, err, n-1-i)
        end
    end
    return psi_err
end

function error_generator(Err, psi::Array{Complex{Float64},2}, p::Float64)
    n = Int(log2(length(psi)))
    psi_err = copy(psi)
    for i=0:n-1
        if rand()< p
            err = rand(Err)
            apply_unitary!(psi_err, err, n-1-i)
        end
    end
    return psi_err
end

stabilizerGroupeOp = Array[ sigma_z, sigma_z, sigma_z, sigma_z,
                            sigma_z, sigma_z, sigma_x, sigma_x]
stabilizerGroupeIdx = Array[[1 0],[2 1],[4 3],[5 4],[7 6],[8 7],[5 4 3 2 1 0],[8 7 6 5 4 3]]

stabilizerCorrectionBitFlipEv = Array[
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
stabilizerCorrectionBitFlipIdx = [0;1;2;3;4;5;6;7;8]

stabilizerCorrectionPhaseFlipEv = Array[[-1;1], [-1;-1], [1;-1]]
stabilizerCorrectionPhaseFlipIdx = Array[[0 1 2], [3 4 5], [6 7 8]];


function error_correction(psi::Array{Complex{Float64},1})
    results = zeros(Int64,length(stabilizerGroupeIdx))

    for i in 1:8
        psi_test = copy(psi)
        for j in 1:length(stabilizerGroupeIdx[i])
            apply_unitary!(psi_test, stabilizerGroupeOp[i], stabilizerGroupeIdx[i][j])
        end
        if psi_test == psi
            results[i] = 1
        elseif psi_test == -psi
            results[i] = -1
        end
    end

    psi_corr = copy(psi)
    for i in 1:9
        if results[1:6] == stabilizerCorrectionBitFlipEv[i]
            apply_unitary!(psi_corr, sigma_x, stabilizerCorrectionBitFlipIdx[i])
        end
    end
    for i in 1:3
        if results[7:8] == stabilizerCorrectionPhaseFlipEv[i]
            for j in 1:3
                apply_unitary!(psi_corr, sigma_z, stabilizerCorrectionPhaseFlipIdx[i][j])
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

function count_coincidence_error_correction(p::Array{Float64,1}, case::AbstractString, N=10000)
    coincidences = SharedArray(Float64, length(p))
    n = 1
    @sync @parallel for j in 1:length(p)
        for i=1:N
            psi = random_state(2)
            if case == "correction"
                psi_enc = encoder(psi)
                psi_enc_err = error_generator(Array[sigma_x], psi_enc, p[j])
                psi_enc_err_err = error_generator(Array[sigma_z], psi_enc_err, p[j])
                psi_enc_corrected = error_correction(psi_enc_err_err)

                if norm(psi - psi_enc_corrected[1:2]) < 1e-6
                    coincidences[j] += 1.
                elseif norm(psi - psi_enc_corrected[1:2] - 2) < 1e-6
                    coincidences[j] += 1.
                end
            elseif case == "no_encoded"
                psi_err = error_generator(Array[sigma_x], psi, p[j])
                psi_enc_err = error_generator(Array[sigma_z], psi_err, p[j])
                if psi == psi_enc_err
                    coincidences[j] += 1.
                end
            end
        end
    end
    return coincidences/N
end


function run_code(N::Int64)
    println("Inicia")

    p = 0.1*collect(0:0.005:1);
    coincidencias_qec = count_coincidence_error_correction(p, "correction", N);
    coincidencias_e = count_coincidence_error_correction(p, "no_encoded", 10000);

    println("Inicia grafica")
    scatter(p, coincidencias_qec, c = "red")
    scatter(p, coincidencias_e, c = "blue")
    xlim([0,maximum(p)]);

    println("Se guarda la imagen")
    savefig("/Users/diego/Documents/Proyecto-Temas/Imagen.png")
    println("Termino")

end


end
